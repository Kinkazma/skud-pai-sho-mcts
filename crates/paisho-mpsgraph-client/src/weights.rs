//! PSW1/PSW2 immutable inference weights, without Adam or checkpoint files.
//! Packet: PWGT0001; four u32 network dimensions (trunk, blocks, embedding,
//! value hidden); f32 epsilon; u64 source step; u32 array count; arrays of
//! (u32 UTF-8 name length, name, u32 rank, rank*u64 dimensions, u64 count,
//! count*f32 values); SHA-256 of every preceding packet byte. All numbers LE.
use sha2::{Digest, Sha256};

use crate::{read_frame, write_frame, MpsGraphClientError, MpsGraphProcess};

pub const MAXIMUM_WEIGHT_PAYLOAD_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSnapshot {
    payload: Vec<u8>,
    content_sha256: [u8; 32],
    training_step: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightImportAck {
    pub request_id: u64,
    pub training_step: u64,
    pub content_sha256: [u8; 32],
}

fn invalid(message: impl Into<String>) -> MpsGraphClientError {
    MpsGraphClientError::WeightsWire(message.into())
}

impl WeightSnapshot {
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
    pub const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }
    pub const fn training_step(&self) -> u64 {
        self.training_step
    }

    /// Verifies packet identity and binary layout. The importing model checks exact
    /// network configuration and parameter names/shapes before assigning variables.
    pub fn from_payload(payload: Vec<u8>) -> Result<Self, MpsGraphClientError> {
        if payload.len() < 72 || payload.len() > MAXIMUM_WEIGHT_PAYLOAD_BYTES - 12 {
            return Err(invalid("invalid weight packet length"));
        }
        let end = payload.len() - 32;
        let content_sha256: [u8; 32] = Sha256::digest(&payload[..end]).into();
        if payload[end..] != content_sha256 {
            return Err(invalid("weight checksum mismatch"));
        }
        let mut reader = Reader {
            bytes: &payload[..end],
            offset: 0,
        };
        if reader.take(8)? != b"PWGT0001" {
            return Err(invalid("invalid weight packet magic"));
        }
        for _ in 0..4 {
            if reader.u32()? == 0 {
                return Err(invalid("zero network dimension"));
            }
        }
        let epsilon = f32::from_bits(reader.u32()?);
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("invalid normalization epsilon"));
        }
        let training_step = reader.u64()?;
        let count = reader.u32()?;
        if count == 0 {
            return Err(invalid("empty weights"));
        }
        let mut names = std::collections::HashSet::new();
        for _ in 0..count {
            let length = reader.u32()? as usize;
            let name = std::str::from_utf8(reader.take(length)?)
                .map_err(|_| invalid("invalid weight name"))?;
            if name.is_empty() || !names.insert(name) {
                return Err(invalid("empty or duplicate weight name"));
            }
            let rank = reader.u32()?;
            if rank == 0 {
                return Err(invalid("zero weight rank"));
            }
            let mut elements = 1_u64;
            for _ in 0..rank {
                let dimension = reader.u64()?;
                if dimension == 0 {
                    return Err(invalid("zero weight dimension"));
                }
                elements = elements
                    .checked_mul(dimension)
                    .ok_or_else(|| invalid("weight shape overflow"))?;
            }
            if reader.u64()? != elements {
                return Err(invalid("weight element count mismatch"));
            }
            let bytes = elements
                .checked_mul(4)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| invalid("weight byte count overflow"))?;
            for chunk in reader.take(bytes)?.chunks_exact(4) {
                let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
                if !value.is_finite() {
                    return Err(invalid("non-finite weight"));
                }
            }
        }
        if reader.offset != end {
            return Err(invalid("trailing weight bytes"));
        }
        Ok(Self {
            payload,
            content_sha256,
            training_step,
        })
    }
}

impl MpsGraphProcess {
    pub fn export_weights(
        &mut self,
        request_id: u64,
    ) -> Result<WeightSnapshot, MpsGraphClientError> {
        let mut request = b"PSW1".to_vec();
        request.extend_from_slice(&request_id.to_le_bytes());
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &request,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            MAXIMUM_WEIGHT_PAYLOAD_BYTES,
        )?;
        let body = response_body(&response, request_id, b"PSWR")?;
        WeightSnapshot::from_payload(body.to_vec())
    }

    pub fn import_weights(
        &mut self,
        request_id: u64,
        snapshot: &WeightSnapshot,
    ) -> Result<WeightImportAck, MpsGraphClientError> {
        let mut request = Vec::with_capacity(12 + snapshot.payload.len());
        request.extend_from_slice(b"PSW2");
        request.extend_from_slice(&request_id.to_le_bytes());
        request.extend_from_slice(snapshot.payload());
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &request,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            1_048_576,
        )?;
        decode_ack(&response, request_id, snapshot)
    }
}

fn decode_ack(
    response: &[u8],
    request_id: u64,
    snapshot: &WeightSnapshot,
) -> Result<WeightImportAck, MpsGraphClientError> {
    let body = response_body(response, request_id, b"PSWA")?;
    if body.len() != 40 {
        return Err(invalid("invalid weight ack length"));
    }
    let training_step = u64::from_le_bytes(body[..8].try_into().expect("eight bytes"));
    let content_sha256: [u8; 32] = body[8..].try_into().expect("digest length");
    if training_step != snapshot.training_step || content_sha256 != snapshot.content_sha256 {
        return Err(invalid("weight import ack identity mismatch"));
    }
    Ok(WeightImportAck {
        request_id,
        training_step,
        content_sha256,
    })
}

fn response_body<'a>(
    response: &'a [u8],
    id: u64,
    magic: &[u8; 4],
) -> Result<&'a [u8], MpsGraphClientError> {
    let mut reader = Reader {
        bytes: response,
        offset: 0,
    };
    let actual = reader.take(4)?;
    if reader.u64()? != id {
        return Err(invalid("weight response request id mismatch"));
    }
    if actual == b"PSWE" {
        let length = reader.u32()? as usize;
        let message = String::from_utf8_lossy(reader.take(length)?).into_owned();
        return Err(invalid(message));
    }
    if actual != magic {
        return Err(invalid("invalid weight response magic"));
    }
    Ok(&response[12..])
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], MpsGraphClientError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| invalid("weight length overflow"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| invalid("truncated weights"))?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, MpsGraphClientError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, MpsGraphClientError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Metal service test: run with benchmark_with_training_paused wrapper"]
    fn service_weight_round_trip() {
        use crate::{NetworkPreset, OptimizationLevel, ServiceConfiguration};
        use paisho_core::{BasicFlower, Position, StandardSetup};
        use paisho_model::{InferenceExampleV1, InferenceRequestV1};
        let service = std::env::var_os("PAISHO_WEIGHTS_TEST_SERVICE")
            .expect("PAISHO_WEIGHTS_TEST_SERVICE must name the built Swift service");
        let configuration = ServiceConfiguration {
            executable: service.into(),
            preset: NetworkPreset::Micro,
            batch_size: 1,
            legal_action_capacity: 128,
            inference_slots: 1,
            optimization: OptimizationLevel::Level1,
            seed: 701,
            checkpoint: None,
        };
        let mut source = MpsGraphProcess::launch(configuration.clone()).unwrap();
        let mut target_configuration = configuration;
        target_configuration.batch_size = 2;
        target_configuration.seed = 709;
        let mut target = MpsGraphProcess::launch(target_configuration).unwrap();
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let example = InferenceExampleV1::from_position(&position).unwrap();
        let request = InferenceRequestV1::from_examples(1, vec![example.clone()], 128).unwrap();
        let target_request =
            InferenceRequestV1::from_examples(2, vec![example.clone(), example], 128).unwrap();
        let expected = source.infer(&request).unwrap();
        let _ = target.infer(&target_request).unwrap();
        let snapshot = source.export_weights(3).unwrap();
        let pid = target.process_id();
        target.import_weights(4, &snapshot).unwrap();
        let actual = target.infer(&target_request).unwrap();
        assert_eq!(target.process_id(), pid);
        assert_eq!(target.export_weights(5).unwrap(), snapshot);
        assert_eq!(actual.outputs().len(), 2);
        for output in actual.outputs() {
            for (a, b) in output
                .policy_probabilities()
                .iter()
                .zip(expected.outputs()[0].policy_probabilities())
            {
                assert!((a - b).abs() <= 1.0e-5);
            }
        }
        assert!(source.shutdown().unwrap().success());
        assert!(target.shutdown().unwrap().success());
    }
    fn fixture() -> Vec<u8> {
        let mut packet = b"PWGT0001".to_vec();
        for v in [1_u32, 1, 1, 1, 1.0e-5_f32.to_bits()] {
            packet.extend(v.to_le_bytes());
        }
        packet.extend(91_u64.to_le_bytes());
        packet.extend(1_u32.to_le_bytes());
        packet.extend(1_u32.to_le_bytes());
        packet.push(b'w');
        packet.extend(1_u32.to_le_bytes());
        packet.extend(2_u64.to_le_bytes());
        packet.extend(2_u64.to_le_bytes());
        packet.extend(0.25_f32.to_le_bytes());
        packet.extend((-0.5_f32).to_le_bytes());
        packet.extend_from_slice(&Sha256::digest(&packet));
        packet
    }
    #[test]
    fn packet_identity_corruption_and_ack() {
        let packet = fixture();
        let snapshot = WeightSnapshot::from_payload(packet.clone()).unwrap();
        assert_eq!(snapshot.training_step(), 91);
        assert_eq!(snapshot.clone().payload(), packet);
        let mut corrupt = packet.clone();
        corrupt[45] ^= 1;
        assert!(WeightSnapshot::from_payload(corrupt).is_err());
        for n in 0..packet.len() {
            assert!(WeightSnapshot::from_payload(packet[..n].to_vec()).is_err());
        }
        let mut ack = b"PSWA".to_vec();
        ack.extend(7_u64.to_le_bytes());
        ack.extend(91_u64.to_le_bytes());
        ack.extend(snapshot.content_sha256());
        assert_eq!(decode_ack(&ack, 7, &snapshot).unwrap().training_step, 91);
        assert!(decode_ack(&ack, 8, &snapshot).is_err());
        ack[20] ^= 1;
        assert!(decode_ack(&ack, 7, &snapshot).is_err());
        fn send<T: Send>() {}
        send::<WeightSnapshot>();
    }
}

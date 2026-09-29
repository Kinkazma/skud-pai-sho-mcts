//! PSG1 JSON transition protocol inside the existing eight-byte length frame.
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::MpsGraphClientError;

pub const MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCycleSchedulerV1 {
    pub learning_rate: f32,
    pub completed_steps: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCycleRandomStateV1 {
    pub name: String,
    pub state: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCycleProgressV1 {
    pub generation: u64,
    pub replay_index: u64,
    #[serde(rename = "replaySnapshotSHA256")]
    pub replay_snapshot_sha256: String,
    pub scheduler: TrainingCycleSchedulerV1,
    pub random_states: Vec<TrainingCycleRandomStateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingCycleRequestV1 {
    pub request_id: u64,
    pub expected_training_step: u64,
    // Required wire key: None is explicitly serialized as JSON null.
    pub previous_snapshot_sha256: Option<String>,
    pub next_progress: TrainingCycleProgressV1,
}

fn wire_error(message: impl Into<String>) -> MpsGraphClientError {
    MpsGraphClientError::TrainingCycleWire(message.into())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl TrainingCycleRequestV1 {
    /// Validate the same local preconditions as Swift; the service owns checks
    /// against its previous snapshot, generation and live training step.
    pub fn encode_payload(&self) -> Result<Vec<u8>, MpsGraphClientError> {
        let progress = &self.next_progress;
        if self
            .previous_snapshot_sha256
            .as_deref()
            .is_some_and(|value| !valid_digest(value))
            || !valid_digest(&progress.replay_snapshot_sha256)
        {
            return Err(wire_error(
                "snapshot SHA-256 must be 64 lowercase hexadecimal characters",
            ));
        }
        if progress.replay_index != 0 {
            return Err(wire_error("new generation replay index must be zero"));
        }
        if progress.scheduler.completed_steps != self.expected_training_step {
            return Err(wire_error(
                "scheduler completed steps differ from expected training step",
            ));
        }
        if !progress.scheduler.learning_rate.is_finite() || progress.scheduler.learning_rate <= 0.0
        {
            return Err(wire_error("learning rate must be finite and positive"));
        }
        let mut names = HashSet::new();
        if progress.random_states.is_empty()
            || progress
                .random_states
                .iter()
                .any(|state| state.name.is_empty() || !names.insert(&state.name))
        {
            return Err(wire_error(
                "random states must have nonempty unique names and cannot be empty",
            ));
        }
        let mut payload = b"PSG1".to_vec();
        payload.extend(serde_json::to_vec(self).map_err(|error| wire_error(error.to_string()))?);
        if payload.len() > MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES {
            return Err(wire_error("training cycle request exceeds 65536 bytes"));
        }
        Ok(payload)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct TrainingCycleResponseV1 {
    pub request_id: u64,
    pub training_step: u64,
    pub progress: TrainingCycleProgressV1,
}

impl TrainingCycleResponseV1 {
    pub fn decode_payload(
        payload: &[u8],
        request: &TrainingCycleRequestV1,
    ) -> Result<Self, MpsGraphClientError> {
        if payload.len() > MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES {
            return Err(wire_error("training cycle response exceeds 65536 bytes"));
        }
        if let Some(body) = payload.strip_prefix(b"PSGE") {
            #[derive(Deserialize)]
            struct ServiceError {
                request_id: u64,
                message: String,
            }
            let error: ServiceError =
                serde_json::from_slice(body).map_err(|error| wire_error(error.to_string()))?;
            if error.request_id != request.request_id {
                return Err(wire_error("training cycle error request ID mismatch"));
            }
            return Err(wire_error(format!(
                "service rejected training cycle: {}",
                error.message
            )));
        }
        let body = payload
            .strip_prefix(b"PSGR")
            .ok_or_else(|| wire_error("invalid training cycle response magic"))?;
        let response: Self =
            serde_json::from_slice(body).map_err(|error| wire_error(error.to_string()))?;
        if response.request_id != request.request_id {
            return Err(wire_error("training cycle response request ID mismatch"));
        }
        if response.training_step != request.expected_training_step {
            return Err(wire_error("training cycle response training step mismatch"));
        }
        // Typed f32 decoding compares the actual scheduler value across JSON spellings.
        // Serde accepts additional properties; only the protocol identity is compared.
        if response.progress != request.next_progress {
            return Err(wire_error("training cycle response progress mismatch"));
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn request() -> TrainingCycleRequestV1 {
        TrainingCycleRequestV1 {
            request_id: u64::MAX,
            expected_training_step: 1,
            previous_snapshot_sha256: None,
            next_progress: TrainingCycleProgressV1 {
                generation: 2,
                replay_index: 0,
                replay_snapshot_sha256: "ab".repeat(32),
                scheduler: TrainingCycleSchedulerV1 {
                    learning_rate: 0.001,
                    completed_steps: 1,
                },
                random_states: vec![TrainingCycleRandomStateV1 {
                    name: "sampler".into(),
                    state: u64::MAX,
                }],
            },
        }
    }

    fn success(request: &TrainingCycleRequestV1) -> Value {
        json!({"request_id": request.request_id, "training_step": request.expected_training_step, "progress": request.next_progress})
    }

    fn payload(magic: &[u8], body: Value) -> Vec<u8> {
        let mut bytes = magic.to_vec();
        bytes.extend(serde_json::to_vec(&body).unwrap());
        bytes
    }

    #[test]
    fn request_has_exact_keys_required_null_and_existing_frame() {
        let request = request();
        let bytes = request.encode_payload().unwrap();
        assert_eq!(&bytes[..4], b"PSG1");
        let body: Value = serde_json::from_slice(&bytes[4..]).unwrap();
        assert_eq!(
            body,
            json!({
                "request_id": u64::MAX, "expected_training_step": 1, "previous_snapshot_sha256": null,
                "next_progress": {"generation": 2, "replayIndex": 0, "replaySnapshotSHA256": "ab".repeat(32),
                "scheduler": {"learningRate": 0.001_f64, "completedSteps": 1},
                    "randomStates": [{"name": "sampler", "state": u64::MAX}]}
            })
        );
        let mut frame = Vec::new();
        crate::write_frame(&mut frame, &bytes).unwrap();
        assert_eq!(&frame[..8], &(bytes.len() as u64).to_le_bytes());
        assert_eq!(
            crate::read_frame(
                &mut std::io::Cursor::new(frame),
                MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES
            )
            .unwrap(),
            bytes
        );
    }

    #[test]
    fn swift_float_spelling_and_extra_properties_are_accepted() {
        let request = request();
        let mut response = success(&request);
        response["progress"]["scheduler"]["learningRate"] = json!(0.0010000000474974513_f64);
        response["extra"] = json!(true);
        response["progress"]["extra"] = json!("ignored");
        let decoded =
            TrainingCycleResponseV1::decode_payload(&payload(b"PSGR", response), &request).unwrap();
        assert_eq!(decoded.progress, request.next_progress);
    }

    #[test]
    fn mismatched_response_identity_is_rejected() {
        let request = request();
        for (pointer, value) in [
            ("/request_id", json!(1)),
            ("/training_step", json!(2)),
            ("/progress/generation", json!(3)),
            ("/progress/replayIndex", json!(1)),
            ("/progress/replaySnapshotSHA256", json!("cd".repeat(32))),
            ("/progress/scheduler/learningRate", json!(0.002)),
            ("/progress/scheduler/completedSteps", json!(2)),
            ("/progress/randomStates/0/name", json!("other")),
            ("/progress/randomStates/0/state", json!(0)),
        ] {
            let mut response = success(&request);
            *response.pointer_mut(pointer).unwrap() = value;
            assert!(
                TrainingCycleResponseV1::decode_payload(&payload(b"PSGR", response), &request)
                    .is_err(),
                "{pointer}"
            );
        }
    }

    #[test]
    fn errors_malformed_payloads_and_size_limits() {
        let request = request();
        let error = TrainingCycleResponseV1::decode_payload(
            &payload(
                b"PSGE",
                json!({"request_id": u64::MAX, "message": "previousSnapshotMismatch"}),
            ),
            &request,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("service rejected training cycle: previousSnapshotMismatch"));
        let error = TrainingCycleResponseV1::decode_payload(
            &payload(b"PSGE", json!({"request_id": 1, "message": "failure"})),
            &request,
        )
        .unwrap_err();
        assert!(error.to_string().contains("request ID mismatch"));
        for bytes in [
            b"PSG".to_vec(),
            b"OTHER".to_vec(),
            b"PSGR{".to_vec(),
            b"PSGR\xff".to_vec(),
            vec![0; MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES + 1],
        ] {
            assert!(TrainingCycleResponseV1::decode_payload(&bytes, &request).is_err());
        }
        let mut oversized = request.clone();
        oversized.next_progress.random_states[0].name =
            "x".repeat(MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES);
        assert!(oversized.encode_payload().is_err());
    }

    #[test]
    fn request_rejects_invalid_transition_fields() {
        let base = request();
        let mut invalid = Vec::new();
        let mut changed = base.clone();
        changed.next_progress.replay_index = 1;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.next_progress.scheduler.completed_steps = 2;
        invalid.push(changed);
        let mut changed = base.clone();
        changed.previous_snapshot_sha256 = Some("AB".repeat(32));
        invalid.push(changed);
        let mut changed = base.clone();
        changed.next_progress.replay_snapshot_sha256 = "bad".into();
        invalid.push(changed);
        for rate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut changed = base.clone();
            changed.next_progress.scheduler.learning_rate = rate;
            invalid.push(changed);
        }
        let mut changed = base.clone();
        changed.next_progress.random_states.clear();
        invalid.push(changed);
        let mut changed = base.clone();
        changed.next_progress.random_states[0].name.clear();
        invalid.push(changed);
        let mut changed = base.clone();
        changed
            .next_progress
            .random_states
            .push(changed.next_progress.random_states[0].clone());
        invalid.push(changed);
        for request in invalid {
            assert!(request.encode_payload().is_err());
        }
    }
}

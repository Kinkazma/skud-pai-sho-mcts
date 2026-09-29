//! Serialize independent parameter ranges, retaining the exact artifact JSON hash.
use super::*;
use sha2::{Digest, Sha256};

fn pieces(a: &MicroArtifact, parallel: &cpu::Ordered) -> (Vec<u8>, Vec<Vec<u8>>, usize) {
    let skeleton = MicroArtifact {
        sequence_memory: a.sequence_memory.clone(),
        schema: a.schema.clone(),
        feature_schema: a.feature_schema.clone(),
        parameters: vec![],
        updates: a.updates,
        provenance: a.provenance.clone(),
    };
    let bytes = serde_json::to_vec(&skeleton).expect("validated artifact serializes");
    // The declared field follows feature_schema and precedes updates. Strings
    // escape quotes, and SequenceMemorySpec contains no parameter object.
    const MARKER: &[u8] = b",\"parameters\":[],\"updates\":";
    let at = bytes.windows(MARKER.len()).position(|v| v == MARKER)
        .expect("artifact parameter serialization contract") + b",\"parameters\":[".len();
    let ranges: Vec<_> = a.parameters.chunks(8192).map(<[f64]>::to_vec).collect();
    let chunks = parallel.map_owned(ranges, |p|p.len(), |p| serde_json::to_vec(p).expect("parameters serialize"));
    (bytes, chunks, at)
}

pub(super) fn identity(a: &MicroArtifact, parallel: &cpu::Ordered) -> String {
    if a.parameters.len() < 32768 { return a.identity(); }
    let (bytes, chunks, at) = pieces(a, parallel);
    let mut hash = Sha256::new();
    hash.update(&bytes[..at]);
    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 { hash.update(b","); }
        hash.update(&chunk[1..chunk.len()-1]);
    }
    hash.update(&bytes[at..]);
    format!("{:x}", hash.finalize())
}

#[doc(hidden)]
pub fn benchmark(path: &Path, output: &Path) -> Result<()> {
    // No bank load is needed to hash an already serialized artifact.
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(path)?)?;
    let pools = cpu::build_search_pools(10, 5, None)?;
    let parallel = cpu::Ordered::new(&pools);
    let expected = artifact.identity();
    if identity(&artifact, &parallel) != expected { return Err(invalid("identity mismatch")); }
    let (bytes, chunks, at) = pieces(&artifact, &parallel);
    let mut joined = bytes[..at].to_vec();
    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 { joined.push(b','); }
        joined.extend_from_slice(&chunk[1..chunk.len()-1]);
    }
    joined.extend_from_slice(&bytes[at..]);
    if joined != serde_json::to_vec(&artifact)? { return Err(invalid("identity bytes mismatch")); }
    let mut trials = vec![];
    for threaded in [false, true, true, false] {
        let t = Instant::now();
        for _ in 0..128 {
            let actual = if threaded { identity(&artifact, &parallel) } else { artifact.identity() };
            assert_eq!(actual, expected);
            std::hint::black_box(actual);
        }
        trials.push(serde_json::json!({"parallel":threaded,"identities":128,"seconds":t.elapsed().as_secs_f64()}));
    }
    let report = serde_json::json!({"bytes_exact":true,"sha256":expected,"parameters":artifact.parameters.len(),"trials":trials});
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunk_boundaries_metadata_and_float_bits_keep_json_and_identity() {
        let pools = cpu::build_search_pools(4, 2, None).unwrap();
        let parallel = cpu::Ordered::new(&pools);
        for n in [0, 1, 8191, 8192, 8193, 32767, 32768, 32769, 292363] {
            let a = MicroArtifact {
                sequence_memory: None,
                schema: "quoted\"parameters\":[],\"updates\":\n".into(),
                feature_schema: "é\\\u{0}".into(),
                parameters: (0..n).map(|i| [0.,-0.,f64::MIN_POSITIVE,f64::from_bits(1),f64::MAX,-0.3,f64::NAN,f64::INFINITY][i%8]).collect(),
                updates: u64::MAX,
                provenance: serde_json::json!({"parameters":[],"updates":123,"nested":{"parameters":[]}}),
            };
            let (bytes, chunks, at) = pieces(&a, &parallel);
            let mut joined = bytes[..at].to_vec();
            for (i, c) in chunks.iter().enumerate() {
                if i > 0 { joined.push(b','); }
                joined.extend_from_slice(&c[1..c.len()-1]);
            }
            joined.extend_from_slice(&bytes[at..]);
            assert_eq!(joined, serde_json::to_vec(&a).unwrap(), "length {n}");
            assert_eq!(identity(&a, &parallel), a.identity());
        }
    }
}

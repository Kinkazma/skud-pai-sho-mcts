use paisho_ai::sequence_source;
use sha2::{Digest,Sha256};
#[test]
fn relocated_source_keeps_original_exclusion_identity() {
    for run in ["private-machine/campaign-a", "another-machine/campaign-b"] {
        for game in ["1", "999999", "reanalysis-7"] {
            let source=format!("{run}/{game}");
            let hash=format!("{:x}",Sha256::digest(source.as_bytes()));
            let relocated=format!("portable-source/{hash}/{game}");
            assert_eq!(sequence_source(&source),sequence_source(&relocated));
        }
    }
}
#[test]
fn malformed_portable_names_keep_normal_hash_semantics() {
    for source in ["portable-source/not-a-hash/3", "portable-source/abc", "human/record"] {
        let hash=Sha256::digest(source.as_bytes());
        assert_eq!(sequence_source(source),u64::from_le_bytes(hash[..8].try_into().unwrap()));
    }
}

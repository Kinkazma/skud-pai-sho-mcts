use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId};
use paisho_model::encode_action_v1;
use paisho_replay::{
    PolicyTargetV1, ReplayCodecV1Error, ReplayDatasetV1, ReplayDecisionV1, ReplayDigestV1,
    ReplayGameV1, ReplayShardReferenceV1, ReplayShardV1, ReplayShardV1Error, ReplaySnapshotV1,
    ReplaySnapshotV1Error, REPLAY_ENCODING_SCHEMA_V1,
};
use sha2::{Digest, Sha256};

const REPORTED_RING: &str = include_str!("../../paisho-core/tests/fixtures/reported-ring-v1.psr");
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "paisho-replay-profiles-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn reported_game(profile: RuleProfileId) -> ReplayGameV1 {
    let source = REPORTED_RING.parse::<GameRecord>().unwrap();
    let record = if profile == source.rules() {
        source
    } else {
        source.replay_prefix_with_rules(profile).unwrap().0
    };
    let position = record.initial_position();
    let agent = ReplayDigestV1::from_bytes([41; 32]);
    let action = encode_action_v1(record.actions()[0], position.to_move()).unwrap();
    let decision = ReplayDecisionV1::new(0, PolicyTargetV1::one_hot(agent, action).unwrap());
    ReplayGameV1::new(
        if profile == RuleProfileId::SkudPaiSho2022 {
            1
        } else {
            2
        },
        agent,
        agent,
        record,
        vec![decision],
    )
    .unwrap()
}

#[test]
fn reported_ring_replays_and_labels_each_profile_independently() {
    let legacy = reported_game(RuleProfileId::SkudPaiSho2022);
    let corrected = reported_game(RuleProfileId::SkudPaiSho2022V2);
    assert_eq!(legacy.record().actions().len(), 76);
    assert_eq!(corrected.record().actions().len(), 74);
    assert_eq!(legacy.outcome(), GameOutcome::Win(Player::Host));
    assert_eq!(corrected.outcome(), GameOutcome::Win(Player::Guest));
    // The first decision belongs to Guest: labels must follow the pinned rules.
    assert_eq!(
        legacy
            .materialize_training_example(0)
            .unwrap()
            .terminal_return(),
        -1.0
    );
    assert_eq!(
        corrected
            .materialize_training_example(0)
            .unwrap()
            .terminal_return(),
        1.0
    );
    assert_eq!(
        legacy.materialize_training_examples().unwrap()[0].terminal_return(),
        -1.0
    );
}

#[test]
fn shards_and_snapshots_round_trip_both_effective_profiles() {
    let directory = Directory::new();
    for (index, profile) in [
        RuleProfileId::SkudPaiSho2022,
        RuleProfileId::SkudPaiSho2022V2,
    ]
    .into_iter()
    .enumerate()
    {
        let shard = ReplayShardV1::new(index as u64, vec![reported_game(profile)]).unwrap();
        let name = format!("{index}.psrbuf");
        let path = directory.0.join(&name);
        shard.write_new(&path).unwrap();
        let bytes = fs::read(&path).unwrap();
        let restored = ReplayShardV1::read(&path).unwrap();
        assert_eq!(restored, shard);
        assert_eq!(restored.rule_profile(), profile);
        let copy = directory.0.join(format!("copy-{index}.psrbuf"));
        restored.write_new(&copy).unwrap();
        assert_eq!(fs::read(copy).unwrap(), bytes);

        let snapshot =
            ReplaySnapshotV1::new(vec![
                ReplayShardReferenceV1::from_shard(&name, &shard).unwrap()
            ])
            .unwrap();
        let path = directory.0.join(format!("{index}.psrsnap"));
        snapshot.write_new(&path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains(&format!("rules\t{profile}\n")));
        let restored = ReplaySnapshotV1::read(&path).unwrap();
        assert_eq!(restored, snapshot);
        assert_eq!(restored.rule_profile(), profile);
        let dataset = ReplayDatasetV1::from_snapshot(&restored, &directory.0).unwrap();
        assert_eq!(dataset.rule_profile(), profile);
        assert_eq!(
            restored.verify_directory(&directory.0).unwrap(),
            restored.verify_directory_integrity(&directory.0).unwrap()
        );
    }
}

#[test]
fn mixed_profiles_are_rejected_by_shards_and_snapshots() {
    let legacy = reported_game(RuleProfileId::SkudPaiSho2022);
    let corrected = reported_game(RuleProfileId::SkudPaiSho2022V2);
    assert!(matches!(
        ReplayShardV1::new(0, vec![legacy.clone(), corrected.clone()]),
        Err(ReplayShardV1Error::Codec(
            ReplayCodecV1Error::RuleProfileMismatch { .. }
        ))
    ));
    let legacy = ReplayShardV1::new(0, vec![legacy]).unwrap();
    let corrected = ReplayShardV1::new(1, vec![corrected]).unwrap();
    assert!(matches!(
        ReplaySnapshotV1::new(vec![
            ReplayShardReferenceV1::from_shard("legacy.psrbuf", &legacy).unwrap(),
            ReplayShardReferenceV1::from_shard("corrected.psrbuf", &corrected).unwrap(),
        ]),
        Err(ReplaySnapshotV1Error::RuleProfileMismatch { .. })
    ));
}

#[test]
fn decoded_shard_checks_game_profile_even_with_a_valid_checksum() {
    let directory = Directory::new();
    let shard = ReplayShardV1::new(0, vec![reported_game(RuleProfileId::SkudPaiSho2022)]).unwrap();
    let path = directory.0.join("mismatch.psrbuf");
    shard.write_new(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    bytes.truncate(bytes.len() - 32);
    // Change only the shard header; the embedded record keeps its original V1.
    let profile_offset = 8 + 4 + 4 + REPLAY_ENCODING_SCHEMA_V1.len();
    let profile = RuleProfileId::SkudPaiSho2022V2.as_str().as_bytes();
    let old_length = RuleProfileId::SkudPaiSho2022.as_str().len();
    let replacement = (profile.len() as u32)
        .to_le_bytes()
        .into_iter()
        .chain(profile.iter().copied())
        .collect::<Vec<_>>();
    bytes.splice(profile_offset..profile_offset + 4 + old_length, replacement);
    let checksum = Sha256::digest(&bytes);
    bytes.extend_from_slice(&checksum);
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        ReplayShardV1::read(&path),
        Err(ReplayShardV1Error::Codec(
            ReplayCodecV1Error::RuleProfileMismatch {
                expected: RuleProfileId::SkudPaiSho2022V2,
                actual: RuleProfileId::SkudPaiSho2022,
                ..
            }
        ))
    ));
}

#[test]
fn snapshot_checks_actual_shard_profile_in_full_and_fast_verification() {
    let directory = Directory::new();
    let shard = ReplayShardV1::new(0, vec![reported_game(RuleProfileId::SkudPaiSho2022)]).unwrap();
    let name = "legacy.psrbuf";
    let digest = shard.write_new(&directory.0.join(name)).unwrap();
    let incorrect =
        ReplayShardReferenceV1::with_rules(name, 0, digest, 1, 1, RuleProfileId::SkudPaiSho2022V2)
            .unwrap();
    let snapshot = ReplaySnapshotV1::new(vec![incorrect]).unwrap();
    for result in [
        snapshot.verify_directory(&directory.0),
        snapshot.verify_directory_integrity(&directory.0),
    ] {
        assert!(matches!(
            result,
            Err(ReplaySnapshotV1Error::RuleProfileMismatch {
                expected: RuleProfileId::SkudPaiSho2022V2,
                actual: RuleProfileId::SkudPaiSho2022,
                ..
            })
        ));
    }
}

#[test]
fn empty_historical_snapshot_preserves_original_bytes_and_digest() {
    let directory = Directory::new();
    let body = format!(
        "PAISHO-REPLAY-SNAPSHOT\t1\nencoding\t{REPLAY_ENCODING_SCHEMA_V1}\nrules\tskud-pai-sho-2022-03-14\nshards\t0\n"
    );
    let digest = ReplayDigestV1::from_bytes(Sha256::digest(body.as_bytes()).into());
    let text = format!("{body}sha256\t{digest}\n");
    let source = directory.0.join("old.psrsnap");
    fs::write(&source, &text).unwrap();
    let snapshot = ReplaySnapshotV1::read(&source).unwrap();
    assert_eq!(snapshot.rule_profile(), RuleProfileId::SkudPaiSho2022);
    assert_eq!(snapshot.digest(), digest);
    let copy = directory.0.join("copy.psrsnap");
    snapshot.write_new(&copy).unwrap();
    assert_eq!(fs::read_to_string(copy).unwrap(), text);
}

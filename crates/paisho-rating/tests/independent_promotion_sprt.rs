use paisho_rating::{
    evaluate_promotion_sprt, PentanomialCounts, PromotionSprtConfig,
    PENTANOMIAL_SPRT_REFERENCE_COMMIT, PENTANOMIAL_SPRT_REFERENCE_PATH,
};

#[test]
fn rust_llr_matches_the_pinned_official_fishtest_reference() {
    let mut lines = include_str!("fixtures/official_pentanomial_sprt.tsv").lines();
    assert_eq!(
        lines.next(),
        Some("PAISHO-PENTANOMIAL-SPRT-OFFICIAL-REFERENCE\t1")
    );
    let expected_commit = format!("source_commit\t{PENTANOMIAL_SPRT_REFERENCE_COMMIT}");
    let expected_path = format!("source_path\t{PENTANOMIAL_SPRT_REFERENCE_PATH}");
    assert_eq!(lines.next(), Some(expected_commit.as_str()));
    assert_eq!(lines.next(), Some(expected_path.as_str()));
    assert_eq!(lines.next(), Some("counts\telo0\telo1\tllr"));

    for line in lines {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 4);
        let raw_counts = fields[0]
            .split(',')
            .map(|field| field.parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        let counts = PentanomialCounts::new(raw_counts.try_into().unwrap());
        let config = PromotionSprtConfig::new(
            fields[1].parse().unwrap(),
            fields[2].parse().unwrap(),
            0.05,
            0.05,
        )
        .unwrap();
        let actual = evaluate_promotion_sprt(counts, config)
            .unwrap()
            .log_likelihood_ratio;
        let expected = fields[3].parse::<f64>().unwrap();
        assert!(
            (actual - expected).abs() <= 2.0e-10,
            "expected {expected:.15}, got {actual:.15} for {}",
            fields[0]
        );
    }
}

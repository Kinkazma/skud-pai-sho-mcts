use super::*;
use paisho_core::*;

fn graph() -> MicroPieceGraph {
    let r: GameRecord = include_str!(
        "../../../paisho-train/examples/gen5_structured_corpus/fixtures/cross_owner_lotus.psr"
    )
    .parse()
    .unwrap();
    let p = r.replay().unwrap();
    MicroPieceGraph::extract(&p, Player::Host)
}

#[test]
fn all_pieces_and_borrowed_lotus_keep_their_owners() {
    let g = graph();
    let mut isolated = 0;
    for (i, n) in g.nodes.iter().enumerate() {
        assert_eq!(n.features[..12].iter().sum::<f64>(), 1.);
        assert_eq!(n.features[12], f64::from(n.owner == g.perspective));
        if !g.messages.iter().any(|m| m.source == i) {
            isolated += 1;
        }
    }
    assert!(isolated > 0);
    assert!(g
        .messages
        .iter()
        .any(|m| m.features[0] == 1. && g.nodes[m.source].owner != g.perspective));
    for m in &g.messages {
        assert!(g.messages.iter().any(|n| n.source == m.destination
            && n.destination == m.source
            && n.features[0] == m.features[0]
            && n.features[1] == m.features[1]
            && n.features[2] == -m.features[2]
            && n.features[3] == -m.features[3]));
    }
}

#[test]
fn encoder_is_invariant_to_node_and_edge_storage_order() {
    let g = graph();
    let mut permuted = g.clone();
    permuted.nodes.reverse();
    for e in &mut permuted.messages {
        e.source = g.nodes.len() - 1 - e.source;
        e.destination = g.nodes.len() - 1 - e.destination;
    }
    permuted.messages.reverse();
    let net = MicroGraphEncoder::seeded(73);
    for mode in [
        MicroGraphPropagation::IndependentPieces,
        MicroGraphPropagation::TwoHarmonyRounds,
    ] {
        let a = net.encode(&g, mode);
        let b = net.encode(&permuted, mode);
        for i in 0..MICRO_GRAPH_OUTPUTS {
            assert!((a[i] - b[i]).abs() < 1e-12);
        }
    }
}

#[test]
fn isolated_piece_has_an_effect_and_empty_graph_is_finite() {
    let mut g = graph();
    let i = (0..g.nodes.len())
        .find(|&i| !g.messages.iter().any(|e| e.source == i))
        .unwrap();
    let net = MicroGraphEncoder::seeded(17);
    let mode = MicroGraphPropagation::TwoHarmonyRounds;
    let a = net.encode(&g, mode);
    // Perturb the feature of an existing isolated piece, without changing edges.
    g.nodes[i].features[14] += 0.125;
    let b = net.encode(&g, mode);
    assert!(a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-7));
    g.nodes.clear();
    g.messages.clear();
    assert_eq!(net.encode(&g, mode), [0.; MICRO_GRAPH_OUTPUTS]);
    assert_eq!(
        net.gradient(&g, mode, &[1.; MICRO_GRAPH_OUTPUTS]),
        vec![0.; MICRO_GRAPH_PARAMETERS]
    );
}

#[test]
fn shared_two_round_gradient_matches_every_parameter_finite_difference() {
    let g = graph();
    let net = MicroGraphEncoder::seeded(31);
    let dy = std::array::from_fn(|i| (i as f64 - 11.) / 17.);
    let node_dy: Vec<[f64; H]> = (0..g.nodes.len())
        .map(|i| std::array::from_fn(|j| ((i + 3 * j) % 11) as f64 / 19.))
        .collect();
    for mode in [
        MicroGraphPropagation::IndependentPieces,
        MicroGraphPropagation::TwoHarmonyRounds,
    ] {
        let grad = net.gradient_with_nodes(&g, mode, &dy, &node_dy).unwrap();
        for k in 0..MICRO_GRAPH_PARAMETERS {
            let loss = |delta: f64| {
                let mut w = net.parameters().to_vec();
                w[k] += delta;
                let m = MicroGraphEncoder::from_parameters(w).unwrap();
                m.encode(&g, mode)
                    .iter()
                    .zip(dy)
                    .map(|(x, d)| x * d)
                    .sum::<f64>()
                    + m.encode_nodes(&g, mode)
                        .iter()
                        .zip(&node_dy)
                        .map(|(h, d)| h.iter().zip(d).map(|(h, d)| h * d).sum::<f64>())
                        .sum::<f64>()
            };
            let numeric = (loss(1e-6) - loss(-1e-6)) / 2e-6;
            assert!(
                (numeric - grad[k]).abs() < 1e-7,
                "{mode:?} parameter {k}: {numeric} != {}",
                grad[k]
            );
        }
    }
}

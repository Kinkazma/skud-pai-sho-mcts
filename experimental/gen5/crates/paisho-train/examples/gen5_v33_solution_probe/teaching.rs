use super::*;
use data::Data;
fn select(d: &Data, mi: usize, train: bool) -> Vec<usize> {
    let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, t) in d.teachers.iter().enumerate().filter(|(_, t)| t.model == mi) {
        groups.entry(&t.group).or_default().push(i);
    }
    let groups: Vec<_> = groups
        .values()
        .enumerate()
        .filter(|(i, _)| (i % 2 == 0) == train)
        .map(|(_, v)| v)
        .collect();
    let mut result = vec![];
    for k in 0..1000 {
        for g in &groups {
            if let Some(i) = g.get(k) {
                result.push(*i);
                if result.len() == 32 {
                    return result;
                }
            }
        }
    }
    result
}
fn metrics(m: &MicroModel, ids: &[usize], d: &Data, values: &BTreeMap<usize, Vec<f64>>) -> Value {
    let mut kl = 0.;
    let mut eq = 0.;
    let mut rawq = 0.;
    let mut top = 0;
    for &i in ids {
        let t = &d.teachers[i];
        let p = prior(m, &t.ex);
        let coupled = micro_softmax(
            &p.iter()
                .zip(&values[&i])
                .enumerate()
                .map(|(k, (p, v))| {
                    if t.excluded[k] {
                        -1e100
                    } else {
                        p.max(1e-300).ln() + 16. * v
                    }
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        kl +=
            t.ex.policy
                .iter()
                .zip(&coupled)
                .filter(|(t, _)| **t > 0.)
                .map(|(t, p)| t * (t.ln() - p.max(1e-300).ln()))
                .sum::<f64>();
        eq += coupled.iter().zip(&t.q).map(|(p, q)| p * q).sum::<f64>();
        rawq += p.iter().zip(&t.q).map(|(p, q)| p * q).sum::<f64>();
        top += usize::from(best(&coupled) == best(&t.ex.policy));
    }
    json!({"n":ids.len(),"coupled_target_kl":kl/ids.len() as f64,"expected_teacher_q":eq/ids.len() as f64,"raw_expected_teacher_q":rawq/ids.len() as f64,"teacher_top_matches":top})
}
pub fn run(d: &Data) -> Result<Value> {
    let mut fits = vec![];
    let mut selections = vec![];
    for mi in [0, 4] {
        let train = select(d, mi, true);
        let test = select(d, mi, false);
        let train_groups: std::collections::BTreeSet<_> =
            train.iter().map(|&i| &d.teachers[i].group).collect();
        assert!(test
            .iter()
            .all(|&i| !train_groups.contains(&d.teachers[i].group)));
        let values: BTreeMap<_, _> = train
            .iter()
            .chain(&test)
            .map(|&i| {
                (
                    i,
                    data::successors(&d.models[mi], &d.teachers[i].position)
                        .iter()
                        .map(|s| s.value(&d.models[mi]))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        selections.push(json!({"model":mi,"train":train.iter().map(|&i|json!({"game":d.teachers[i].game,"decision":d.teachers[i].decision,"source":d.teachers[i].group})).collect::<Vec<_>>(),"test":test.iter().map(|&i|json!({"game":d.teachers[i].game,"decision":d.teachers[i].decision,"source":d.teachers[i].group})).collect::<Vec<_>>()}));
        for seed in [17, 31, 53] {
            for mode in 0..3 {
                let corrected = mode == 1;
                let mut model = d.models[mi].clone();
                let retention = |model: &MicroModel| {
                    let mut scores = vec![];
                    for exclude_training_sources in [false, true] {
                        let mut n = 0;
                        let mut wins = 0;
                        let mut mass = 0.;
                        for r in d
                            .proofs
                            .iter()
                            .filter(|r| r.group == "outside" && r.ex.value == 1.)
                        {
                            if exclude_training_sources
                                && train_groups.iter().any(|s| s.as_str() == r.source)
                            {
                                continue;
                            }
                            let p = prior(model, &r.ex);
                            n += 1;
                            wins += usize::from(r.union[best(&p)]);
                            mass += p
                                .iter()
                                .zip(&r.union)
                                .filter(|(_, v)| **v)
                                .map(|(p, _)| p)
                                .sum::<f64>();
                        }
                        scores.push(json!({"exclude_training_sources":exclude_training_sources,"n":n,"wins":wins,"mass":mass/n as f64}));
                    }
                    scores
                };
                let retention_before = retention(&model);
                let mut rng = StableRng::new(seed);
                let mut history = vec![
                    json!({"epoch":0,"train":metrics(&model,&train,d,&values),"test":metrics(&model,&test,d,&values)}),
                ];
                let started = Instant::now();
                for epoch in 1..=24 {
                    let mut order = train.clone();
                    for i in (1..order.len()).rev() {
                        let j = rng.index(i + 1);
                        order.swap(i, j);
                    }
                    for indices in order.chunks(8) {
                        let batch: Vec<_> = indices
                            .iter()
                            .map(|&i| {
                                if corrected {
                                    &d.teachers[i].corrected
                                } else {
                                    &d.teachers[i].ex
                                }
                            })
                            .collect();
                        let rate = 0.02 * batch.len() as f64 / 64.;
                        if mode == 2 {
                            // g(T)-g(P_current) = J_logits^T(P_current-T).
                            // Successor values are detached, fixed and reused here.
                            let mut gradient = vec![0.; model.parameters().len()];
                            for &i in indices {
                                let teacher = &d.teachers[i];
                                let raw = prior(&model, &teacher.ex);
                                let mut current = teacher.ex.clone();
                                current.policy = micro_softmax(
                                    &raw.iter()
                                        .zip(&values[&i])
                                        .enumerate()
                                        .map(|(k, (p, v))| {
                                            if teacher.excluded[k] {
                                                -1e100
                                            } else {
                                                p.max(1e-300).ln() + 16. * v
                                            }
                                        })
                                        .collect::<Vec<_>>(),
                                )?;
                                let gt = model.loss_gradient(&teacher.ex)?.1;
                                let gp = model.loss_gradient(&current)?.1;
                                for ((g, a), b) in gradient.iter_mut().zip(gt).zip(gp) {
                                    *g += (a - b) / indices.len() as f64;
                                }
                            }
                            model = step(&model, &gradient, rate);
                        } else {
                            model.train_batch_inline(&batch, rate, 0.)?;
                        }
                    }
                    if [8, 24].contains(&epoch) {
                        history.push(json!({"epoch":epoch,"train":metrics(&model,&train,d,&values),"test":metrics(&model,&test,d,&values)}));
                    }
                }
                assert!(model
                    .parameters()
                    .iter()
                    .zip(d.models[mi].parameters())
                    .enumerate()
                    .filter(|(i, _)| value_parameter(*i))
                    .all(|(_, (a, b))| a.to_bits() == b.to_bits()));
                fits.push(json!({"model":mi,"seed":seed,"mode":(["legacy_raw","corrected_raw","coupled_loss"][mode]),"corrected_raw_target":corrected,"history":history,"seconds":started.elapsed().as_secs_f64(),"value_parameters_exact":true,"proof_retention_before":retention_before,"proof_retention_after":retention(&model)}));
            }
        }
    }
    let cache = cache_probe(d)?;
    Ok(
        json!({"fits":fits,"selections":selections,"cache":cache,"scope":"Finite policy-only learning on frozen search targets; source-disjoint test within already-known campaign data, not game-strength validation."}),
    )
}
fn signature(r: &MicroSearchReport) -> Value {
    json!({"actions":r.actions.iter().map(ToString::to_string).collect::<Vec<_>>(),"selected":r.selected_index,"policy":r.policy_target,"prior":r.priors,"search_prior":r.search_priors,"q":r.values,"visits":r.visits,"new_visits":r.new_visits,"forced":r.new_forced_visits,"pruned":r.pruned_visits,"proof":r.proven_value,"action_proofs":r.proven_action_values,"value":r.network_value,"state":r.state,"features":r.action_features.as_ref(),"simulations":r.simulations})
}
fn cache_probe(d: &Data) -> Result<Value> {
    let ids: Vec<_> = select(d, 4, true).into_iter().take(16).collect();
    let model = Arc::new(d.models[4].clone());
    let mut baseline = vec![];
    let mut runs = vec![];
    for (round, cached) in [false, true, true, false].into_iter().enumerate() {
        let mut bank: BTreeMap<usize, Arc<MicroSearchReport>> = BTreeMap::new();
        let mut hits = 0;
        let mut searches = 0;
        let t = Instant::now();
        let mut signatures = vec![];
        for &i in ids.iter().chain(&ids) {
            let result = if cached && bank.contains_key(&i) {
                hits += 1;
                bank[&i].clone()
            } else {
                let mut session = MicroMctsSession::new(model.clone());
                session.set_root_value_strength(16.)?;
                let r = Arc::new(session.search_with_options(
                    &d.teachers[i].position,
                    512,
                    None,
                    MicroSearchOptions {
                        proof_search: true,
                        forced_playout_strength: 0.,
                        gumbel_scale: 0.,
                        dirichlet_fraction: 0.,
                        seed: 20260912,
                        ..Default::default()
                    },
                )?);
                searches += 1;
                if cached {
                    bank.insert(i, r.clone());
                }
                r
            };
            signatures.push(signature(&result));
        }
        let seconds = t.elapsed().as_secs_f64();
        if round == 0 {
            baseline = signatures;
        } else {
            assert_eq!(baseline, signatures);
        }
        runs.push(json!({"cached":cached,"requests":ids.len()*2,"actual_searches":searches,"cache_hits":hits,"seconds":seconds}));
    }
    Ok(
        json!({"abba":runs,"request_outputs_exact":true,"scope":"16 fixed roots, two identical requests each, immutable model/options and no external proof change; cold and warm hits included, no full-game throughput claim"}),
    )
}

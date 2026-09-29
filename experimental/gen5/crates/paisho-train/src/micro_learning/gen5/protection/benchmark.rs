//! Frozen ordered SGD and finite consolidation controls for parallel scheduling.
use super::*;
pub(in crate::micro_learning::gen5) fn run(model: &MicroModel, rows: &[Arc<MicroExample>], pools: &[Arc<rayon::ThreadPool>]) -> Result<serde_json::Value> {
    let original=original_references(model,rows)?;
    let mut sequential=Protection::new(model,rows.to_vec())?;
    assert!(original.iter().flatten().zip(sequential.gradients.iter().flatten()).all(|(x,y)|x.to_bits()==y.to_bits()),"V35 original reference gradient differs");
    let mut parallel=Protection::new(model,rows.to_vec())?;
    parallel.enable_parallel(pools);
    let mut a=model.clone();let mut b=model.clone();
    let batch=rows.iter().cycle().take(64).map(AsRef::as_ref).collect::<Vec<_>>();
    let mut seconds=[0.;2];
    for _ in 0..8 {
        let mut gradient=vec![0.;a.parameters().len()];
        for e in &batch {
            let (_,g)=a.loss_gradient(e).map_err(invalid)?;
            for (x,y) in gradient.iter_mut().zip(g) {*x+=y/batch.len() as f64;}
        }
        for (g,w) in gradient.iter_mut().zip(a.parameters()) {*g+=1e-6*w;}
        let projected=original_project(&gradient,&sequential.gradients,&[0.;4]).ok_or_else(||invalid("original projection"))?;
        let norm=dot(&projected,&projected).sqrt();let scale=if norm>10. {10./norm}else{1.};
        let expected:Vec<_>=a.parameters().iter().zip(projected).map(|(w,g)|w-0.001*scale*g).collect();
        let t=Instant::now();sequential.train(&mut a,&batch,0.001,1e-6)?;seconds[0]+=t.elapsed().as_secs_f64();
        let t=Instant::now();parallel.train(&mut b,&batch,0.001,1e-6)?;seconds[1]+=t.elapsed().as_secs_f64();
        assert!(a.parameters().iter().zip(expected).all(|(x,y)|x.to_bits()==y.to_bits()),"original V35 protected step differs");
        assert!(a.parameters().iter().zip(b.parameters()).all(|(x,y)|x.to_bits()==y.to_bits()),"ordered protected update differs");
    }
    sequential.observe(rows);parallel.observe(rows);
    let t=Instant::now();sequential.consolidate(&mut a)?;let serial_consolidate=t.elapsed().as_secs_f64();
    let t=Instant::now();parallel.consolidate(&mut b)?;let parallel_consolidate=t.elapsed().as_secs_f64();
    assert!(a.parameters().iter().zip(b.parameters()).all(|(x,y)|x.to_bits()==y.to_bits()),"parallel consolidation differs");
    assert_eq!(sequential.last,parallel.last);
    assert!(sequential.gradients.iter().flatten().zip(parallel.gradients.iter().flatten()).all(|(x,y)|x.to_bits()==y.to_bits()),"reference gradients differ");
    Ok(serde_json::json!({"ordered_updates":8,"batch_examples":64,"all_parameter_bits_exact":true,"reference_gradient_bits_exact":true,"consolidation_decision_exact":true,"sequential_train_seconds":seconds[0],"parallel_train_seconds":seconds[1],"sequential_consolidation_seconds":serial_consolidate,"parallel_consolidation_seconds":parallel_consolidate}))
}

fn original_references(m: &MicroModel, rows: &[Arc<MicroExample>]) -> Result<Vec<Vec<f64>>> {
    let mut gs = vec![vec![0.; m.parameters().len()]; 4];
    let mut ns = [0usize; 4];
    for r in rows {
        let mut ex = r.as_ref().clone();
        ex.policy_weight = 0.;
        ex.value_weight = 1.;
        let k = (ex.value as i8 + 1) as usize + 1;
        let (_, g) = m.loss_gradient(&ex).map_err(invalid)?;
        ns[k] += 1;
        for (a, b) in gs[k].iter_mut().zip(g) {
            *a += b;
        }
        if ex.value == 1. && !ex.policy.is_empty() {
            let p = prior(m, &ex)?;
            let mass: f64 = p
                .iter()
                .zip(&ex.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, _)| p)
                .sum();
            if mass <= 0. {
                return Err(invalid("zero verified policy mass"));
            }
            ex.policy = p
                .iter()
                .zip(&ex.policy)
                .map(|(p, t)| if *t > 0. { p / mass } else { 0. })
                .collect();
            ex.policy_weight = 1.;
            ex.value_weight = 0.;
            let (_, g) = m.loss_gradient(&ex).map_err(invalid)?;
            ns[0] += 1;
            for (a, b) in gs[0].iter_mut().zip(g) {
                *a += b;
            }
        }
    }
    for k in 0..4 {
        for g in &mut gs[k] {
            *g /= ns[k].max(1) as f64;
        }
    }
    Ok(gs)
}


// Euclidean projection onto <=5 gradient halfspaces, enumerating active sets.
fn original_project(g: &[f64], refs: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = refs.len();
    let gram: Vec<Vec<f64>> = refs
        .iter()
        .map(|a| refs.iter().map(|b| dot(a, b)).collect())
        .collect();
    let b: Vec<_> = refs
        .iter()
        .zip(rhs)
        .map(|(a, rhs)| dot(a, g) - rhs)
        .collect();
    let mut answer = None;
    let mut distance = f64::INFINITY;
    for mask in 0..1usize << n {
        let ids: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let k = ids.len();
        let mut a: Vec<Vec<f64>> = ids
            .iter()
            .map(|&i| {
                ids.iter()
                    .map(|&j| gram[i][j])
                    .chain(std::iter::once(-b[i]))
                    .collect()
            })
            .collect();
        let mut singular = false;
        for c in 0..k {
            let pivot = (c..k)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, pivot);
            let v = a[c][c];
            if v.abs() < 1e-14 {
                singular = true;
                break;
            }
            for j in c..=k {
                a[c][j] /= v;
            }
            for i in 0..k {
                if i != c {
                    let f = a[i][c];
                    for j in c..=k {
                        a[i][j] -= f * a[c][j];
                    }
                }
            }
        }
        if singular {
            continue;
        }
        let mut lambdas = vec![0.; n];
        for (i, &r) in ids.iter().enumerate() {
            lambdas[r] = a[i][k];
        }
        if lambdas.iter().any(|v| *v < -1e-10) {
            continue;
        }
        if (0..n).any(|i| b[i] + dot(&gram[i], &lambdas) < -1e-10) {
            continue;
        }
        let dist = (0..n)
            .map(|i| lambdas[i] * dot(&gram[i], &lambdas))
            .sum::<f64>();
        if dist < distance {
            distance = dist;
            let mut out = g.to_vec();
            for (l, r) in lambdas.iter().zip(refs) {
                for (x, v) in out.iter_mut().zip(r) {
                    *x += l * v;
                }
            }
            answer = Some(out);
        }
    }
    answer
}


// Frozen pre-repair implementation retained exclusively for controlled cost and
// behavior comparisons. It is never called by the campaign training path.
impl Protection {
    fn consolidate_legacy_benchmark(&mut self, model: &mut MicroModel) -> Result<()> {
        let t = paisho_platform::training_time::now();
        self.checks += 1;
        let (limits, old_choices) = losses(&self.anchor, &self.rows, self.parallel.as_ref())?;
        let mut candidate = model.clone();
        let fresh = self.fresh.iter().cloned().collect::<Vec<_>>();
        let old_fresh = fresh_loss(&self.anchor, &fresh, self.parallel.as_ref())?;
        let learned_fresh = fresh_loss(model, &fresh, self.parallel.as_ref())?;
        let mut iterations = 0;
        let mut choice_constraints = 0;
        let mut final_choices = old_choices.clone();
        let mut success = false;
        let mut final_losses = limits;
        let mut final_fresh = learned_fresh;
        for i in 0..=6 {
            let (current, choices) = losses(&candidate, &self.rows, self.parallel.as_ref())?;
            final_losses = current;
            final_choices = choices.clone();
            iterations = i;
            let finite = current.iter().zip(limits).all(|(a, b)| *a <= b + 1e-9);
            let retained = old_choices.iter().zip(&choices).all(|(a, b)| !*a || *b);
            if finite && retained {
                final_fresh = fresh_loss(&candidate, &fresh, self.parallel.as_ref())?;
                let ceiling = if learned_fresh < old_fresh {
                    old_fresh - 0.05 * (old_fresh - learned_fresh)
                } else {
                    learned_fresh
                };
                success = final_fresh <= ceiling + 1e-12;
                break;
            }
            if i == 6 {
                break;
            }
            let mut refs = references(&candidate, &self.rows, self.parallel.as_ref())?;
            self.reference_evaluations += self.rows.len();
            let mut rhs = current
                .iter()
                .zip(limits)
                .map(|(a, b)| a - b + 1e-9)
                .collect::<Vec<_>>();
            if let Some((gradient, gap)) =
                lost_choice_constraint(&candidate, &self.rows, &old_choices)?
            {
                choice_constraints += 1;
                refs.push(gradient);
                rhs.push(gap);
                self.reference_evaluations += 2;
            }
            let Some(correction) = original_project(&vec![0.; model.parameters().len()], &refs, &rhs) else {
                break;
            };
            candidate = weights(
                &candidate,
                candidate
                    .parameters()
                    .iter()
                    .zip(correction)
                    .map(|(w, g)| w - g)
                    .collect(),
            )?;
        }
        if success {
            *model = candidate;
            self.accepted += 1;
        }
        self.last = serde_json::json!({"accepted":success,"iterations":iterations,"choice_constraints":choice_constraints,"raw_before":old_choices.iter().filter(|x|**x).count(),"raw_after":final_choices.iter().filter(|x|**x).count(),"lost_choices":old_choices.iter().zip(&final_choices).filter(|(a,b)|**a && !**b).count(),"reference_before":limits,"reference_after":final_losses,"fresh_anchor":old_fresh,"fresh_before":learned_fresh,"fresh_after":final_fresh,"constraint_scope":"mean class losses and individual raw verified choices; publication separately verifies coupled choices"});
        // The learner continues if consolidation fails; it does not become an actor.
        // The publication guard still compares every actor to its accepted predecessor.
        self.anchor = model.clone();
        self.gradients = references(model, &self.rows, self.parallel.as_ref())?;
        self.gram=projection::gram(&self.gradients);
        self.reference_evaluations += self.rows.len();
        self.seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
        Ok(())
    }
}

/// Compare a frozen anchor/candidate transaction in ABBA order. Loading,
/// construction, result serialization and hashing are outside the timed region.
pub(in crate::micro_learning::gen5) fn transaction(
    anchor: &MicroModel,
    candidate: &MicroModel,
    rows: &[Arc<MicroExample>],
    fresh: &[Arc<MicroExample>],
    pools: &[Arc<rayon::ThreadPool>],
) -> Result<serde_json::Value> {
    let mut results = Vec::new();
    let mut corrected_parameters = None;
    for corrected in [false, true, true, false] {
        let mut p = Protection::new(anchor, rows.to_vec())?;
        p.enable_parallel(pools);
        p.observe(fresh);
        let mut model = candidate.clone();
        let started = Instant::now();
        if corrected { p.consolidate(&mut model)?; }
        else { p.consolidate_legacy_benchmark(&mut model)?; }
        let seconds = started.elapsed().as_secs_f64();
        let parameter_bits = model.parameters().iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        if corrected {
            if let Some(expected) = &corrected_parameters {
                assert_eq!(expected, &parameter_bits, "corrected consolidation is deterministic");
            }
            corrected_parameters = Some(parameter_bits);
            if !p.last["accepted"].as_bool().unwrap_or(false) {
                assert!(model.parameters().iter().zip(anchor.parameters()).all(|(a,b)|a.to_bits()==b.to_bits()));
                assert!(p.anchor.parameters().iter().zip(anchor.parameters()).all(|(a,b)|a.to_bits()==b.to_bits()));
            }
        }
        results.push(serde_json::json!({
            "corrected":corrected,"seconds":seconds,"progress":p.progress(),
            "applied_losses":losses(&model,rows,Some(&cpu::Ordered::new(pools)))?.0,
            "parameters_sha256":sha256(&serde_json::to_vec(&model.parameters())?),
            "anchor_sha256":sha256(&serde_json::to_vec(&p.anchor.parameters())?),
            "fresh_examples_retained":p.fresh.len(),
        }));
    }
    Ok(serde_json::json!({"rows":rows.len(),"fresh_rows":fresh.len(),"runs":results,
        "corrected_repeated_bits_exact":true,"production_modified":false}))
}

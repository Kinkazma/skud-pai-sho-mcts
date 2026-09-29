//! Fixed-input timings checked against the unmodified V35 scoring algorithm.
use super::*;
mod reference;
pub fn run(config: &Path, out: &Path) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let o: Options = serde_json::from_slice(&fs::read(config)?)?;
    let a = MicroArtifact::load(&o.model)?;
    let model = a.model()?;
    let resume: serde_json::Value = serde_json::from_slice(&fs::read(o.resume_progress.as_ref().unwrap())?)?;
    let snapshot = Arc::new(Snapshot {
        identity:a.identity(), model:Arc::new(model.clone()), artifact:Some(Arc::new(a)),
        version:resume["version"].as_u64().unwrap(), path:o.model.clone(),
    });
    let mut guard = Guard::open(o.publication_guard.as_ref().unwrap(),out,o.value_policy_strength,snapshot,&resume["publication_guard"])?;
    guard.enable_v2(o.publication_validation.as_ref().unwrap())?;
    let pools=if o.learner_threads>0 {vec![cpu::build_pool(o.learner_threads,None)?.0]} else {cpu::build_search_pools(10,5,None)?};
    guard.enable_parallel(&pools);
    let actor=guard.accepted();
    let mut records=vec![];
    for (panel_index,panel) in std::iter::once(&guard).chain(guard.validation.as_deref()).enumerate() {
        for (name,m) in [("actor",actor.model.as_ref()),("learner",&model)] {
            let expected=reference::measure_cached(&panel.rows,m,panel.beta,true)?;
            for parallel in [false,true,true,false] {
                let t=Instant::now();
                let score=if parallel {panel.evaluate(m)?} else {reference::measure_cached(&panel.rows,m,panel.beta,true)?};
                let seconds=t.elapsed().as_secs_f64();
                if score.raw!=expected.raw || score.coupled!=expected.coupled || score.mass.to_bits()!=expected.mass.to_bits() || score.value_mse.to_bits()!=expected.value_mse.to_bits() {
                    return Err(invalid("V35/reference publication score differs"));
                }
                records.push(serde_json::json!({"panel":panel_index,"rows":panel.rows.len(),"model":name,"parallel":parallel,"seconds":seconds,"exact":true}));
            }
        }
    }
    let cache_checks=verify_variants(&guard,&model)?;
    let protection=protection::benchmark_parallel(&model,&guard.reference_examples(),&pools)?;
    let result=serde_json::json!({"all_scores_bit_exact":true,"cache_invalidation_checks":cache_checks,"records":records,"protection":protection});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

fn verify_variants(guard:&Guard,model:&MicroModel)->Result<usize> {
    let mut checked=0;
    for index in [MICRO_NEURAL_MEMORY_START,4160,MICRO_VALUE_TRUNK,MICRO_DEEP_VALUE_START] {
        if index>=model.parameters().len(){continue;}
        let mut w=model.parameters().to_vec();w[index]+=0.125;
        let mut candidate=MicroModel::from_parameters(w).map_err(invalid)?;
        if let Some(bank)=model.sequence_memory(){candidate=candidate.with_sequence_memory_owned(bank.clone());}
        for panel in std::iter::once(guard).chain(guard.validation.as_deref()) {
            let original=reference::measure_cached(&panel.rows,&candidate,panel.beta,true)?;
            for _ in 0..2 {
                let cached=panel.evaluate(&candidate)?;
                if original.raw!=cached.raw || original.coupled!=cached.coupled || original.mass.to_bits()!=cached.mass.to_bits() || original.value_mse.to_bits()!=cached.value_mse.to_bits() {
                    return Err(invalid("value/policy cache invalidation differs from original"));
                }
                checked+=1;
            }
        }
    }
    Ok(checked)
}

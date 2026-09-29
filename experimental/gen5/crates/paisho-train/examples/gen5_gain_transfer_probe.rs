//! Frozen two-cycle gain-transfer diagnostic; never a runtime campaign.
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a=std::env::args().collect::<Vec<_>>();
    if !(a.len()==4 || (a.len()==5 && a[4]=="--relinearize-after-accepted")) {
        return Err("FROZEN_SELECTION CYCLE NEW_OUTPUT_DIRECTORY [--relinearize-after-accepted]".into());
    }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let run=if a.len()==5 {paisho_train::micro_learning::gen5::probe_gain_transfer_relinearized}
        else {paisho_train::micro_learning::gen5::probe_gain_transfer};
    let report=run(std::path::Path::new(&a[1]),a[2].parse()?,std::path::Path::new(&a[3]))?;
    println!("{}",serde_json::json!({"cycle":report["cycle"],"native_proofs_verified":report["native_proofs_verified"],"diagnostic_only":true}));Ok(())
}

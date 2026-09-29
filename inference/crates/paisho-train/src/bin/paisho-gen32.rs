use paisho_train::gen32::*;
use std::{env, fs, path::Path};
fn main() -> Result<()> {
    let a: Vec<String> = env::args().collect();
    match a.get(1).map(String::as_str){
        Some("compare-panel") if a.len()==9=>compare_panel(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),a[5].parse()?,a[6].parse()?,a[7].parse()?,Path::new(&a[8])),
        Some("compare-learning") if a.len()==8=>compare_learning(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),a[5].parse()?,a[6].parse()?,a[7].parse()?),
        Some("compare-tactics") if a.len()==9=>compare_tactics(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),a[5].parse()?,a[6].parse()?,a[7].parse()?,a[8].parse()?),
        Some("defaults") if a.len()==2=>{println!("{}",serde_json::to_string(&Options::default())?);Ok(())},
        Some("bootstrap") if a.len()>=4=>bootstrap(Path::new(&a[2]),Path::new(&a[3]),a.get(4).map(Path::new)),
        Some("human-fit") if a.len()==5=>human_fit(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4])),
        Some("run") if a.len()==3=>run(serde_json::from_slice(&fs::read(&a[2])?)?),
        Some("compare") if a.len()==8=>compare_seed(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),a[5].parse()?,a[6].parse()?,a[7].parse()?),
        Some("compare") if a.len()==7=>compare(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),a[5].parse()?,a[6].parse()?),
        _=>Err("usage: paisho-gen32 bootstrap PARENT OUTPUT [MEMORY_DIR] | run CONFIG | compare MODEL GEN31 OUT BUDGET PAIRS [SEED] | compare-learning MODEL GEN32 OUT BUDGET PAIRS SEED | compare-panel MODEL REFERENCE OUT BUDGET SEED REFERENCE_SOLVER PANEL".into()),
    }
}

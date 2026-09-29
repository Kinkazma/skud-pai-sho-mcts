//! Latest teaching target for revisited exact prefixes; old evidence is immutable.
use super::*;
use durable::Lesson;
#[derive(Serialize,Deserialize)]
struct Revision {
    rules:String,
    prefix_sha256:String,
    version:u64,
    bundle:PathBuf,
    lesson:Lesson,
    lesson_sha256:String,
}
pub(super) fn store(root:&Path, game:&collector::Played, saved:&[SavedMicroExample], bundle:&Path) -> Result<()> {
    let directory=root.join("revisions"); fs::create_dir_all(&directory)?;
    for s in saved.iter().filter(|s|game.reanalysis || s.decision==game.prefix_decisions+1 || s.correction_priority) {
        let key=sha256(cases::prefix(&game.record,s.decision-1).to_string().as_bytes());
        let path=directory.join(format!("{key}.json"));
        if path.exists() {
            let old:Revision=serde_json::from_slice(&fs::read(&path)?)?;
            if old.version>game.snapshot.version { continue; }
        }
        let lesson=Lesson::from_saved(s);
        let revision=Revision { rules:RULES.to_string(),prefix_sha256:key,version:game.snapshot.version,
            bundle:bundle.to_path_buf(),lesson_sha256:sha256(&serde_json::to_vec(&lesson)?),lesson };
        durable::write_pending(&path,&revision)?;
    }
    Ok(())
}
pub(super) fn lookup(root:&Path,key:&str) -> Result<Option<Lesson>> {
    let path=root.join("revisions").join(format!("{key}.json"));
    if !path.exists() { return Ok(None); }
    let revision:Revision=serde_json::from_slice(&fs::read(path)?)?;
    if revision.rules!=RULES.as_str() || revision.prefix_sha256!=key || sha256(&serde_json::to_vec(&revision.lesson)?)!=revision.lesson_sha256 {
        return Err(invalid("durable revision identity mismatch"));
    }
    Ok(Some(revision.lesson))
}

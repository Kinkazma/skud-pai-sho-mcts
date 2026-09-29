//! Independent per-opponent cases; global admission counts fresh continuations only.
//! No opponent waits for another opponent's case, search or promotion.
use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, RwLock,
};
pub(super) fn group(ticket: usize) -> usize {
    if ticket % 5 < 4 {
        0
    } else {
        (ticket / 5) % 5 + 1
    }
}
fn ordinal(ticket:usize)->usize {if group(ticket)==0{ticket/5*4+ticket%5}else{ticket/25}}
fn initial_game(ticket:usize)->bool {ordinal(ticket)%20==if group(ticket)==0{0}else{1}}
pub(super) fn key(group: usize) -> String {
    if group == 0 {
        "selfplay".into()
    } else {
        format!("3.{group}")
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    actor: usize,
    references: Arc<Vec<Arc<references::Frozen>>>,
    ladders: Arc<Vec<ladder::Shared>>,
    evaluations: Option<Arc<evaluation::Evaluations>>,
    mut states: Vec<cases::State>,
    cases: Arc<Vec<cases::Case>>,
    shared: Arc<RwLock<Arc<Snapshot>>>,
    proofs: durable::Proofs,
    next: Arc<AtomicUsize>,
    count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    tx: mpsc::SyncSender<collector::Played>,
    pool: cpu::Executor,
    o: Options,
    source: String,
    end: Instant,
) {
    let config = o.case_curriculum.as_ref().unwrap();
    let mut pending_cache: Option<((PathBuf, String), GameRecord, GameOutcome)> = None;
    while !stop.load(Ordering::Relaxed) && paisho_platform::training_time::now() < end {
        // Pending feedback belongs to exactly one group and is never reassigned.
        let mut admission=None;
        let selected=states.iter().position(|s|!s.pending.is_empty()).unwrap_or_else(||{let ticket=count.fetch_add(1,Ordering::Relaxed);admission=Some(ticket);group(ticket)});
        let mut state = states[selected].clone();
        if state.rotate_reason.is_some() && state.pending.is_empty() {
            state.advance(o.actors);
        }
        let budget = if selected > 0 {
            Some(ladders[selected - 1].read().unwrap().budget())
        } else {
            None
        };
        if state.pending.is_empty()
            && state.legacy_budget.is_some()
            && state.legacy_budget != budget
        {
            state.advance(o.actors);
        }
        if state.pending.is_empty() {
            state.legacy_budget = budget;
        }
        let case = &cases[state.ticket % cases.len()];
        let id = next.fetch_add(1, Ordering::Relaxed);
        if id >= o.games {
            break;
        }
        let snapshot = shared.read().unwrap().clone();
        if o.learning_loop_repair && selected>0 && admission.is_some_and(|t|ordinal(t)%2==0) {
            let job=evaluations.as_ref().unwrap().reserve(selected-1,&references[selected-1].spec.sha256,budget.unwrap(),snapshot.clone());
            match job {
                Ok(Some(job))=>{
                    let mut measured=o.clone();measured.measurement=true;measured.budgets=vec![(512,1.)];
                    measured.candidate_seat=Some(job.seat);measured.match_reference=Some((references[selected-1].clone(),budget.unwrap()));
                    let mut game=collector::play_from(id,job.snapshot.clone(),job.snapshot,None,&measured,end,&pool,&source,Some(&job.case.record),false,None);
                    game.evaluation=Some(job.attempt);
                    game.case=Some(cases::Attempt{opponent_generation:Some(key(selected)),actor,case:job.case.identity,human_source:job.case.source,zone:0,prefix_decisions:job.case.record.actions().len(),before:state.clone(),after:state,kind:"frozen-measurement".into()});
                    if case_actor::submit(game,&tx,&stop,end).is_none(){break;}continue;
                },
                Ok(None)=>{},
                Err(e)=>{
                    let mut game=collector::play_from(id,snapshot.clone(),snapshot,None,&o,paisho_platform::training_time::now(),&pool,&source,Some(&case.record),false,Some(&proofs));
                    game.error=Some(e.to_string());let _=tx.send(game);break;
                }
            }
        }
        let initial=o.learning_loop_repair && admission.is_some_and(initial_game);
        let initial_record=initial.then(||cases::prefix(&case.record,0));
        let mut options = o.clone();
        let pending = if let Some(index) = state.pending.first() {
            (|| -> Result<Option<GameRecord>> {
                let (path, hash) = state
                    .pending_record
                    .as_ref()
                    .ok_or_else(|| invalid("pending reference case missing"))?;
                if pending_cache
                    .as_ref()
                    .map_or(true, |(key, _, _)| key.0 != *path || key.1 != *hash)
                {
                    let bytes = fs::read(path)?;
                    if sha256(&bytes) != *hash {
                        return Err(invalid("pending reference PSR changed"));
                    }
                    let r: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
                    let outcome=r.replay()?.outcome();
                    pending_cache = Some(((path.clone(), hash.clone()), r, outcome));
                }
                let r = &pending_cache.as_ref().unwrap().1;
                if *index > r.actions().len() {
                    return Err(invalid("pending reference prefix missing"));
                }
                Ok(Some(cases::prefix(&r, *index)))
            })()
        } else {
            Ok(None)
        };
        let pending = match pending {
            Ok(p) => p,
            Err(e) => {
                let mut game = collector::play_from(
                    id,
                    snapshot.clone(),
                    snapshot,
                    None,
                    &options,
                    paisho_platform::training_time::now(),
                    &pool,
                    &source,
                    Some(&case.record),
                    false,
                    Some(&proofs),
                );
                game.error = Some(e.to_string());
                let _ = tx.send(game);
                break;
            }
        };
        let reanalysis = pending.is_some();
        if o.learning_loop_repair && reanalysis {
            let (key,_,outcome)=pending_cache.as_ref().unwrap();
            options.observed_origin=match outcome {
                GameOutcome::Win(Player::Host)=>Some((1.,key.1.clone())),
                GameOutcome::Win(Player::Guest)=>Some((-1.,key.1.clone())),
                GameOutcome::Draw=>Some((0.,key.1.clone())),_=>None,
            };
        }
        let mut after = state.clone();
        if reanalysis {
            options.budgets = vec![(config.reanalysis_budget, 1.)];
            after.pending.remove(0);
        } else if selected > 0 {
            options.match_reference = Some((references[selected - 1].clone(), budget.unwrap()));
            options.candidate_seat = Some(match state.focus.as_deref() {
                Some("H") => Player::Host,
                Some("G") => Player::Guest,
                _ if state.ticket % 2 == 0 => Player::Host,
                _ => Player::Guest,
            });
        }
        let mut game = collector::play_from(
            id,
            snapshot.clone(),
            snapshot,
            None,
            &options,
            end,
            &pool,
            &source,
            Some(pending.as_ref().or(initial_record.as_ref()).unwrap_or(&case.record)),
            reanalysis,
            Some(&proofs),
        );
        if !reanalysis {
            let hash = sha256(game.record.to_string().as_bytes());
            if !initial {after.observe(game.outcome, &hash, config);}
            after.pending = collector::correction_prefixes(&game, config.reanalysis_positions)
                .iter()
                .map(|p| p.actions().len())
                .collect();
            after.pending_record = Some((
                Path::new(&source)
                    .join("games")
                    .join(format!("game-{id:07}.psr")),
                hash,
            ));
        }
        game.case = Some(cases::Attempt {
            opponent_generation: Some(key(selected)),
            actor,
            case: if initial {sha256(initial_record.as_ref().unwrap().to_string().as_bytes())}else{case.identity.clone()},
            human_source: case.source.clone(),
            zone: case.zone,
            prefix_decisions: game.prefix_decisions,
            before: state,
            after: after.clone(),
            kind: if reanalysis {
                "reanalysis"
            } else if initial {
                "initial-game"
            } else {
                "continuation"
            }
            .into(),
        });
        if case_actor::submit(game, &tx, &stop, end).is_none() {
            break;
        }
        states[selected] = after;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn initial_quota_is_five_percent_inside_each_group_and_restarts_exactly() {
        let mut counts=[0;6];let mut initial=[0;6];let mut measured=[0;6];
        for offset in [0,3701] {for t in offset..offset+10000 {let g=group(t);counts[g]+=1;initial[g]+=usize::from(initial_game(t));measured[g]+=usize::from(g>0 && ordinal(t)%2==0);}}
        assert_eq!(counts,[16000,800,800,800,800,800]);assert_eq!(initial,[800,40,40,40,40,40]);assert_eq!(measured,[0,400,400,400,400,400]);
    }
    #[test]
    fn admission_is_eighty_twenty_with_no_cross_opponent_barrier() {
        for offset in [0, 7, 10473] {
            let mut counts = [0; 6];
            for i in offset..offset + 2500 {
                counts[group(i)] += 1;
            }
            assert_eq!(counts, [2000, 100, 100, 100, 100, 100]);
        }
    }
}

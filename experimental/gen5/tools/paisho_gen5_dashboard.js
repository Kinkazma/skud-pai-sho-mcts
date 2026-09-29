// Client-only language changes reuse the current status; they never poll or
// change training. French source strings also cover the static page text.
const english = {
  'Tableau affiché':'Dashboard view','Campagne active':'Active run',
  'Évolution par tranches de dix minutes':'Progress in ten-minute intervals',
  'Campagne actuelle · Gen5 contre Gen3.1 · résultats séparés par budget MCTS.':'Current run · Gen5 vs Gen3.1 · results separated by MCTS budget.',
  'Budget Gen3.1':'Gen3.1 budget',
  'Budget MCTS':'MCTS budget',
  'Adversaire':'Opponent',
  'Gen5 · toutes les références':'Gen5 · all references',
  'Gen5 contre {generation}':'Gen5 vs {generation}',
  'Campagne actuelle · résultats séparés par adversaire et budget MCTS.':'Current run · results separated by opponent and MCTS budget.',

  'Chargement de la mémoire':'Loading replay memory',
  'État du processus conservé':'Process state retained',
  'Reprise directe en RAM':'Resume from resident memory',
  'Aucune évaluation disponible.':'No evaluation available.',
  'Évaluations séparées désactivées dans ce mode. La progression est suivie par les paliers Gen3.1.':'Separate evaluations are disabled in this mode. Progress is tracked through Gen3.1 stages.',

  'Checkpoint durable':'Durable checkpoint',
  'Pause':'Pause', 'Reprendre':'Resume', 'Heures':'Hours', 'Démarrer':'Start',
  'En pause':'Paused',
  'Arrêté':'Stopped',
  'Arrêt et sauvegarde…':'Stopping and saving…',
  'Arrêté · durée restante annulée':'Stopped · remaining time cancelled',
  'Stop annule les heures restantes et conserve une sauvegarde reprenable. Les deux modes poursuivent le même modèle Gen5, un seul à la fois.':'Stop cancels the remaining hours and keeps a resumable checkpoint. Both modes continue the same Gen5 model, one at a time.',
  'Auto-jeu et rejeu Gen5':'Gen5 self-play and case replay',
  'Rejeu contre Gen3.1':'Replay against Gen3.1',
  'Rejeu contre Gen3.1 · trois zones · sans coupure par partie':'Replay against Gen3.1 · three zones · no per-game time cap',
  'Palier Gen3.1 · MCTS {budget} · {wins} victoires / {games} premiers essais sur 100 · objectif 60 victoires':'Gen3.1 stage · MCTS {budget} · {wins} wins / {games} first attempts of 100 · target 60 wins',
  'Dernier lot MCTS {budget} : {wins} V / {draws} N / {losses} D / {unknown} indéterminées':'Last MCTS {budget} batch: {wins} W / {draws} D / {losses} L / {unknown} unresolved',
  'Pause conserve la RAM et le temps restant. Reprise directe si disponible ; sinon, reprise après sauvegarde. Choisissez une durée quand la campagne est en pause ou terminée.':'Pause preserves RAM and remaining time. Resume directly when available, or recover from a saved state. Choose a duration when training is paused or finished.',
  'Préparation de la reprise…':'Preparing training continuation…',
  'En pause · {hours} h restantes':'Paused · {hours} h remaining',
  'Durée : de 1 minute à 24 heures.':'Duration: from 1 minute to 24 hours.',

  "Évolution heure par heure":"Hour-by-hour progress",
  "Tranches horaires fixes · heure de Paris · résultats enregistrés, reprises comprises. L’heure en cours est partielle.":"Fixed hourly intervals · Paris time · recorded results, including resumed runs. The current hour is partial.",
  "Heure":"Hour",
  "Taux de victoire":"Win rate",
  "Taux de victoire calculé parmi les parties terminées de la tranche. Les interruptions restent visibles dans les tentatives.":"Win rate among finished games in this interval. Interrupted games remain included in attempts.",
  "Historique Elo par heure":"Hourly Elo history",
  "Dernière mesure publiée dans chaque heure, sans moyenne ni report de l’heure précédente. Écarts provisoires face à chaque référence fixe ; Gen5 initiale vaut 0 Elo. Les lots et limites peuvent différer : les résultats résolus et indéterminés restent indiqués.":"Last measurement published in each hour, with no averaging or carry-forward. Provisional differences against each fixed reference; initial Gen5 is 0 Elo. Batches and limits may differ: resolved and unresolved results remain visible.",
  "Référence":"Reference",
  "Elo mesuré":"Measured Elo",
  "Mesuré à":"Measured at",
  "V / N / D":"W / D / L",
  "Indéterminées":"Unresolved",
  "Simulations / limite de décisions":"Simulations / decision limit",
  "Chargement de l’historique…":"Loading history…",
  "Historique indisponible : {error}":"History unavailable: {error}",
  "En cours · partielle":"In progress · partial",
  "Aucune mesure":"No measurement",
  "Aucun résultat résolu":"No resolved result",
  "Aucune partie enregistrée.":"No games recorded.",
  "Gen5 initiale":"Initial Gen5",

  'PAI SHO · CONTRÔLE LOCAL':'PAI SHO · LOCAL DASHBOARD',
  'Génération 5':'Generation 5',
  'Protocole Gen3.1 → Gen3.x':'Gen3.1 → Gen3.x protocol',
  'Parties récentes · PSR':'Recent games · PSR',
  'Lecture de la campagne…':'Loading training run…',
  'Production et apprentissage':'Production and learning',
  'Débit Gen5 · fenêtre glissante de dix minutes':'Gen5 throughput · rolling ten-minute window',
  'Débit calculé avec les compteurs natifs, réanalyses exclues. Fenêtre observée de dix minutes maximum ; moyenne de la session en attendant deux relevés.':'Throughput from native counters, excluding reanalyses. Observed window of at most ten minutes; session average until two readings are available.',
  'Répartition des parties':'Game breakdown',
  'Total':'All time',
  'Dernière heure':'Last hour',
  'Résultats cumulés, reprises comprises.':'Cumulative results, including resumed runs.',
  'Résultats enregistrés au cours des 60 dernières minutes, reprises comprises. La fenêtre avance automatiquement.':'Results recorded in the last 60 minutes, including resumed runs. The window moves automatically.',
  'Chargement de la dernière heure…':'Loading the last hour…',
  'Dernière heure indisponible : {error}':'Last hour unavailable: {error}',
  'File':'Queue',
  'Terminées / tentatives':'Finished / attempted',
  'Gen5 gagne':'Gen5 wins',
  'Adversaire gagne':'Opponent wins',
  'Nulles':'Draws',
  'Décisions':'Decisions',
  'Positions neuves apprises':'New positions learned',
  'Relectures':'Replay samples',
  'Relectures déclenchées':'Replay samples triggered',
  'Relectures déclenchées par ces parties, toutes sources confondues.':'Replay samples triggered by these games, from all source queues.',
  'Host et Guest sont des places tirées au sort. Les victoires des confrontations sont attribuées au modèle, quelle que soit sa place. En auto-jeu, deux Gen5 jouent : chaque partie décisive donne une victoire et une défaite au même modèle.':'Host and Guest are randomly assigned seats. Match wins are attributed to the model, regardless of its seat. Self-play uses two Gen5 players: every decisive game is both a win and a loss for the same model.',
  'Évaluations et Elo interne':'Evaluations and internal Elo',
  'En attente d’une évaluation.':'Awaiting an evaluation.',
  'Écarts provisoires contre le modèle initial, séparés des résultats contre Gen3.1. Les checkpoints de poids sont conservés régulièrement ; les jalons de +100 Elo demandent une évaluation complète.':'Provisional differences against the initial model, separate from Gen3.1 results. Weight checkpoints are saved periodically; +100 Elo milestones require a complete evaluation.',
  'Dernières parties sélectionnées':'Latest selected games',
  'Quelques parties récentes, courtes, longues ou nulles. Actualisation toutes les 30 secondes. Toutes les parties originales sont conservées dans la campagne.':'Selected recent, short, long or drawn games, plus recent historical matches. Updated every 30 seconds. All original games are retained in the run.',
  'Partie':'Game',
  'Modèle joué':'Model version played',
  'Place de Gen5':'Gen5 seat',
  'Résultat':'Result',
  'Secondes':'Seconds',
  'Télécharger':'Download',
  'Copier':'Copy',
  'État':'Status',
  'En cours':'Running',
  'Démarrage':'Starting',
  'Terminé':'Completed',
  'Échec':'Failed',
  'Préparation':'Preparing',
  'Version publiée':'Published version',
  'Version apprenante':'Learner version',
  'Fenêtre du débit':'Throughput window',
  'Version sauvegardée':'Saved version',
  'Modèles conservés sur disque':'Models retained on disk',
  'Gen3.1–3.5 · parties / s':'Gen3.1–3.5 · games / s',
  'Gen5 contre Gen3.1–3.5':'Gen5 vs Gen3.1–3.5',
  '80 % auto-jeu · 20 % contre Gen3.1–3.5 · adversaires en RAM':'80% self-play · 20% against Gen3.1–3.5 · resident opponents',
  'Gen{generation} · MCTS {budget} · {wins} victoires / {games} premiers essais sur 100':'Gen{generation} · MCTS {budget} · {wins} wins / {games} first attempts of 100',
  'Objectif indépendant : 60 victoires sur 100, 50 essais par côté.':'Independent target: 60 wins out of 100, 50 attempts per seat.',
  'Gen3.1–3.5 · résultats regroupés par budget MCTS.':'Gen3.1–3.5 · results aggregated by MCTS budget.',
  'Progression suivie séparément pour les cinq adversaires.':'Progress tracked separately for the five opponents.',

  'Mises à jour du modèle':'Model updates',
  'Parties archivées':'Archived games',
  'Rappel effectivement appris':'Recall actually learned',
  'Réserve de rappel en RAM':'Recall reserve in RAM',
  'Gen5 corrigée prête · rappel {percent} % au prochain lancement':'Repaired Gen5 ready · {percent}% recall on the next start',
  ' · Rappel {percent} % de tous les exemples · mises à jour proportionnelles aux exemples':' · Recall {percent}% of all examples · updates proportional to examples',
  'Réanalyses':'Reanalyses', 'Leçons durables':'Durable lesson bundles', 'Preuves vérifiées':'Verified proofs',
  'Rejeu de cas humains · trois zones · sans coupure par partie':'Human-case replay · three zones · no per-game time cutoff',
  'Rejeu Gen5 · cas humains':'Gen5 replay · human cases',
  'Fin prévue':'Scheduled end',
  'Auto-jeu Gen5':'Gen5 self-play',
  'Gen5 · versions retenues':'Gen5 · retained versions',
  'Gen5 contre Gen3.1':'Gen5 vs Gen3.1',
  'Gen5 · parties / s':'Gen5 · games / s',
  'Gen3.1 · parties / s':'Gen3.1 · games / s',
  'Mémoire de positions':'Replay positions',
  'RAM du tampon':'Replay RAM',
  'Exemples humains utilisés':'Human samples used',
  'Capacité réservée historique':'Reserved historical capacity',
  'Actualisation…':'Updating…',
  'Terminée':'Finished',
  'Interrompue':'Interrupted',
  'Interrompue (temps)':'Interrupted (time limit)',
  'Interrompue (limite de coups)':'Interrupted (decision limit)',
  'Interrompue (boucle)':'Interrupted (repetition)',
  'Interrompue (erreur)':'Interrupted (error)',
  'Gen3.1 gagne':'Gen3.1 wins',
  'Version retenue gagne':'Retained version wins',
  'Partie nulle':'Draw',
  'Les deux':'Both',
  'Copie indisponible ; utilisez le téléchargement.':'Copy unavailable; please download the file.',
  'PSR de la partie {id} copié.':'PSR for game {id} copied.',
  'Actualisation : {age} s · PID {pid} · {remaining} minutes restantes':'Updated {age} s ago · PID {pid} · {remaining} minutes remaining',
  '{mode} · {budgets} · {threads} travailleurs CPU, dont {shared} partagés · Coupure auto-jeu {cap} s · Plafond historique {fraction} % de la capacité ; réutilisable par Gen5.':'{mode} · {budgets} · {threads} CPU workers, including {shared} shared · Self-play cutoff {cap} s · Historical cap: {fraction}% of capacity; reusable by Gen5.',
  'Deux Gen5 · {count} parties décisives':'Two Gen5 players · {count} decisive games',
  'Elo interne provisoire · Gen5 vs Gen5 initiale':'Provisional internal Elo · Gen5 vs initial Gen5',
  'Version évaluée':'Evaluated version',
  'Lot évalué':'Evaluation batch',
  'Gen5 initiale · référence fixe':'Initial Gen5 · fixed reference',
  'Gen5 initiale = 0 Elo. Cette comparaison mesure la progression face à sa propre version de départ, sans attendre Gen3.1.':'Initial Gen5 = 0 Elo. This comparison measures progress against its own starting version, without waiting for Gen3.1.',
  'Première évaluation en attente.':'Awaiting the first evaluation.',
  'Évaluation en cours':'Evaluation in progress',
  'Évaluation terminée':'Evaluation finished',
  '{status} · {done} / {total} parties traitées, dont {resolved} résolues.':'{status} · {done} / {total} games processed, including {resolved} resolved.',
  'Évaluations espacées d’au moins {minutes} minutes, sous le plafond CPU historique.':'Evaluations are spaced at least {minutes} minutes apart, subject to the historical CPU cap.',
  'Valeur provisoire · {resolved} / {total} parties résolues contre le modèle initial.':'Provisional value · {resolved} / {total} games resolved against the initial model.',
  'Panel incomplet : trop de résultats manquants pour conclure à un progrès.':'Incomplete panel: too many missing results to conclude that strength improved.',
  'Panel complet ; estimation encore provisoire.':'Complete panel; the estimate remains provisional.',
  'Référence initiale':'Initial reference',
  '{name} : {wins} V / {draws} N / {losses} D / {unknown} indéterminées':'{name}: {wins} W / {draws} D / {losses} L / {unknown} unresolved',
  'Place {seat} gagne (modèle à identifier)':'{seat} seat wins (model unidentified)',
  'Lecture momentanément indisponible : {error}':'Status temporarily unavailable: {error}'
};
const $ = id => document.getElementById(id);
let lang = 'fr';
try { lang = (new URLSearchParams(location.search).get('lang')||localStorage.getItem('paisho-gen5-language')) === 'en' ? 'en' : 'fr'; } catch (_) {}
const locale = () => lang === 'en' ? 'en-GB' : 'fr-FR';
function t(source, values = {}) {
  return (lang === 'en' ? english[source] || source : source).replace(/\{(\w+)\}/g, (match, key) => values[key] ?? match);
}
const fmt = (v, d = 0) => v == null || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString(locale(), {maximumFractionDigits:d});
const esc = s => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const card = (label, value) => '<div class="metric"><small>' + esc(t(label)) + '</small><strong>' + esc(value) + '</strong></div>';
const staticText = [];
const walker = document.createTreeWalker(document.querySelector('main'), NodeFilter.SHOW_TEXT);
while (walker.nextNode()) {
  const node = walker.currentNode, source = node.textContent.trim();
  if (english[source]) staticText.push([node, source]);
}
let latest = null, prev = null, points = [], actionPending = false;
let period = 'total';
try { period = localStorage.getItem('paisho-gen5-period') === 'hour' ? 'hour' : 'total'; } catch (_) {}
function setPeriod(next) {
  period = next === 'hour' ? 'hour' : 'total';
  try { localStorage.setItem('paisho-gen5-period', period); } catch (_) {}
  if (latest) render(latest);
}
function setLanguage(next) {
  lang = next === 'en' ? 'en' : 'fr';
  document.documentElement.lang = lang;
  $('language').textContent = lang === 'en' ? 'Français' : 'English';
  $('language').setAttribute('aria-label', lang === 'en' ? 'Passer en français' : 'Switch to English');
  $('spark').setAttribute('aria-label', t('Débit Gen5 · fenêtre glissante de dix minutes'));
  staticText.forEach(([node, source]) => { node.textContent = t(source); });
  $('copy-status').textContent = '';
  try { localStorage.setItem('paisho-gen5-language', lang); } catch (_) {}
  if (latest) render(latest);
}
$('language').addEventListener('click', () => setLanguage(lang === 'fr' ? 'en' : 'fr'));
async function copy(id) {
  try {
    const response = await fetch('/examples/files/' + id + '.psr');
    if (!response.ok) throw new Error(response.status);
    await navigator.clipboard.writeText(await response.text());
    $('copy-status').textContent = t('PSR de la partie {id} copié.', {id});
  } catch (_) { $('copy-status').textContent = t('Copie indisponible ; utilisez le téléchargement.'); }
}
async function trainingAction(action,hours,mode) {
  if(actionPending)return;
  if(action==='start'&&(!Number.isFinite(hours)||hours<1/60||hours>24)) {
    $('training-action-status').textContent=t('Durée : de 1 minute à 24 heures.');return;
  }
  actionPending=true;
  document.querySelectorAll('.training-controls button').forEach(b=>b.disabled=true);
  $('training-action-status').textContent=t(action==='stop'?'Arrêt et sauvegarde…':'Préparation de la reprise…');
  try {
    const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json','X-Paisho-Control':controlToken},body:JSON.stringify({action,hours,mode})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||response.status);
    actionPending=false;await refresh();
  } catch(error) {
    actionPending=false;
    if(latest)render(latest);
    $('training-action-status').textContent=String(error.message||error);
  }
}
function renderControls(c) {
  const busy=actionPending||c.busy;
  const legacy=c.mode==='gen3-replay';
  $('train-stop').disabled=busy||legacy||(!c.running&&!c.paused);
  $('legacy-stop').disabled=busy||!legacy||(!c.running&&!c.paused);
  $('legacy-pause').disabled=busy||!legacy||!c.running;
  $('legacy-resume').disabled=busy||!legacy||!c.paused||!(c.remaining_seconds>0);
  document.querySelectorAll('.legacy-duration').forEach(b=>b.disabled=busy||c.running||!c.modes_available);
  $('legacy-hours').disabled=busy||c.running||!c.modes_available;
  $('train-pause').disabled=busy||legacy||!c.running;
  $('train-resume').disabled=busy||legacy||!c.paused||!(c.remaining_seconds>0);
  document.querySelectorAll('.train-duration').forEach(b=>b.disabled=busy||c.running);
  $('train-hours').disabled=busy||c.running;
  $('training-action-status').textContent=c.error|| (busy?t(c.busy_action==='stop'?'Arrêt et sauvegarde…':'Préparation de la reprise…'):c.stopped?t('Arrêté · durée restante annulée'):c.paused?t('En pause · {hours} h restantes',{hours:fmt(c.remaining_seconds/3600,2)}):'');
  if(c.prepared_next?.protocol?.structural_repair)$('training-action-status').textContent+=' · '+t('Gen5 corrigée prête · rappel {percent} % au prochain lancement',{percent:fmt(100*c.prepared_next.protocol.recall_fraction)});
}
function resultLabel(g) {
  const unresolved = /^Place (Host|Guest) gagne \(modèle à identifier\)$/.exec(g.result_label || '');
  const generationWin=/^(Gen3\.[1-5]) gagne$/.exec(g.result_label||'');
  if(generationWin)return lang==='en'?generationWin[1]+' wins':g.result_label;
  return unresolved ? t('Place {seat} gagne (modèle à identifier)', {seat:unresolved[1]}) : t(g.result_label || 'Interrompue');
}
function render(s) {
  const control=s.controls||{};
  renderControls(control);
  if(typeof renderGen3==='function')renderGen3();
  if(typeof renderGen3Panels==='function'&&renderGen3Panels())return;
  const p=(control.paused||control.stopped)?{...s.progress,remaining_seconds:control.remaining_seconds}:s.progress, o=s.options, lanes=p.lanes||{}, elapsed=s.session_elapsed_seconds||0, remaining=p.remaining_seconds||0;
  const names={Selfplay:t(o.case_curriculum?'Rejeu Gen5 · cas humains':'Auto-jeu Gen5'),Checkpoint:t('Gen5 · versions retenues'),Historical:t(o.opponents?.length?'Gen5 contre Gen3.1–3.5':'Gen5 contre Gen3.1')};
  const states={running:'En cours',starting:'Démarrage',completed:'Terminé',failed:'Échec'};
  $('main').innerHTML=card('État',t(control.stopped?'Arrêté':control.paused?'En pause':s.initializing?'Chargement de la mémoire':states[s.state.state]||'Préparation'))+(control.paused&&control.memory_resident?card('Mémoire',t(control.resident_resume_available?'Reprise directe en RAM':'État du processus conservé')):'')+card('Version publiée',fmt(p.publication_guard?.accepted_version??p.version))+card('Version apprenante',fmt(p.version))+card('Mises à jour du modèle',fmt(p.updates))+card('Parties archivées',fmt((p.completed||0)-(lanes.Reanalysis?.games||0)))+card('Fin prévue',s.end_local&&!control.stopped?new Date(s.end_local).toLocaleTimeString(locale(),{hour:'2-digit',minute:'2-digit',hour12:false}):'—');
  $('state').textContent=t('Actualisation : {age} s · PID {pid} · {remaining} minutes restantes',{age:fmt(s.progress_age_seconds),pid:s.state.training_pid||'—',remaining:fmt(remaining/60,1)});
  $('time').value=elapsed+remaining?100*elapsed/(elapsed+remaining):0;
  $('learning').innerHTML=card('Gen5 · parties / s',fmt(s.gen5_terminal_per_second,2))+card(o.opponents?.length?'Gen3.1–3.5 · parties / s':'Gen3.1 · parties / s',fmt(s.historical_terminal_per_second,3))+(o.checkpoint_seconds?card('Version sauvegardée',fmt(p.durable_version))+card('Modèles conservés sur disque',fmt(p.checkpoint_models_retained)+' / '+fmt(o.checkpoint_keep||64)):'')+card('Fenêtre du débit',fmt(s.throughput_window?.seconds,1)+' s')+card('Positions neuves apprises',fmt(p.fresh_terminal_used))+card('Mémoire de positions',fmt(p.replay_positions))+card('RAM du tampon',fmt(p.replay_bytes/2**30,2)+' GiB')+card('Exemples humains utilisés',fmt(p.human_used))+card('Capacité réservée historique',fmt(s.historical_reserved_capacity_percent,2)+' %');
  if(o.case_curriculum)$('learning').innerHTML+=card('Réanalyses',fmt(lanes.Reanalysis?.games||0))+card('Leçons durables',fmt(p.durable_bundles))+card('Preuves vérifiées',fmt(p.durable_proofs));
  if(o.structural_repair){
    const q=p.recall_quotas||{}, r=p.durable_recall||{};
    $('learning').innerHTML+=card('Rappel effectivement appris',q.consumed_examples?fmt(100*q.consumed_recall/q.consumed_examples,1)+' %':'—')+card('Réserve de rappel en RAM',fmt(r.cache_positions)+' · '+fmt((r.cache_bytes||0)/2**20,1)+' MiB');
  }
  $('recipe').textContent=t('{mode} · {budgets} · {threads} travailleurs CPU, dont {shared} partagés · Coupure auto-jeu {cap} s · Plafond historique {fraction} % de la capacité ; réutilisable par Gen5.',{mode:(o.mode||'PUCT').toUpperCase(),budgets:JSON.stringify(o.budgets||[]),threads:fmt(o.threads),shared:fmt(o.secondary_threads),cap:fmt(o.game_seconds,2),fraction:fmt(100*o.historical_capacity_fraction,1)});
  if(o.case_curriculum)$('recipe').textContent=t(o.legacy_replay?'Rejeu contre Gen3.1 · trois zones · sans coupure par partie':'Rejeu de cas humains · trois zones · sans coupure par partie');
  if(o.structural_repair)$('recipe').textContent+=t(' · Rappel {percent} % de tous les exemples · mises à jour proportionnelles aux exemples',{percent:fmt(100*o.recall_fraction)});
  const ladder=p.legacy_ladder||{}, entries=ladder.current||[], batch=(ladder.batches||[]).at(-1);
  $('ladder-status').textContent=o.legacy_replay?t('Palier Gen3.1 · MCTS {budget} · {wins} victoires / {games} premiers essais sur 100 · objectif 60 victoires',{budget:[8,32,64,128][ladder.stage||0],wins:entries.filter(e=>e.win).length,games:entries.length})+(batch?' · '+t('Dernier lot MCTS {budget} : {wins} V / {draws} N / {losses} D / {unknown} indéterminées',{budget:batch.budget,wins:batch.wins,draws:batch.draws,losses:batch.games-batch.wins-batch.draws-batch.unknown,unknown:batch.unknown}):''):'';
  if(o.opponents?.length) {
    $('recipe').textContent=t('80 % auto-jeu · 20 % contre Gen3.1–3.5 · adversaires en RAM')+t(' · Rappel {percent} % de tous les exemples · mises à jour proportionnelles aux exemples',{percent:fmt(100*o.recall_fraction)});
    $('ladder-status').style.whiteSpace='pre-line';
    $('ladder-status').textContent=Object.entries(p.opponent_ladders||{}).map(([generation,l])=>t('Gen{generation} · MCTS {budget} · {wins} victoires / {games} premiers essais sur 100',{generation,budget:[8,32,64,128,256,512,1024,2048][l.stage||0],wins:(l.current||[]).filter(e=>e.win).length,games:(l.current||[]).length})).join('\n')+'\n'+t('Objectif indépendant : 60 victoires sur 100, 50 essais par côté.');
  }
  const hour=s.last_hour||{}, hourly=period==='hour';
  $('period-total').setAttribute('aria-pressed', String(!hourly));
  $('period-hour').setAttribute('aria-pressed', String(hourly));
  $('period-note').textContent=hourly?t(hour.error?'Dernière heure indisponible : {error}':hour.ready?'Résultats enregistrés au cours des 60 dernières minutes, reprises comprises. La fenêtre avance automatiquement.':'Chargement de la dernière heure…',{error:hour.error}):t('Résultats cumulés, reprises comprises.');
  $('replay-heading').textContent=t(hourly?'Relectures déclenchées':'Relectures');
  if(hourly&&hour.ready)$('period-note').textContent+=' '+t('Relectures déclenchées par ces parties, toutes sources confondues.');
  const emptyHour={games:0,terminal:0,wins:0,losses:0,draws:0,decisions:0,fresh_used:0,replay_triggered:0,ready:hour.ready};
  const displayed=hourly?Object.fromEntries(Object.keys(lanes).map(k=>[k,(hour.lanes||{})[k]||emptyHour])):lanes;
  $('lanes').innerHTML=hourly&&!hour.ready?'<tr><td colspan="8">'+esc(t(hour.error?'Dernière heure indisponible : {error}':'Chargement de la dernière heure…',{error:hour.error}))+'</td></tr>':Object.entries(displayed).filter(([k])=>k!=='Reanalysis').map(([k,v])=>{
    const r=hourly?v:(s.seat_results||{})[k], cell=x=>'<td>'+esc(x)+'</td>';
    const wins=k==='Selfplay'?'<td colspan="2">'+esc(t('Deux Gen5 · {count} parties décisives',{count:fmt(v.terminal-v.draws)}))+'</td>':cell(r&&r.ready?fmt(r.wins):t('Actualisation…'))+cell(r&&r.ready?fmt(r.losses):t('Actualisation…'));
    return '<tr>'+cell(names[k]||k)+cell(fmt(v.terminal)+' / '+fmt(v.games))+wins+[v.draws,v.decisions,v.fresh_used,hourly?v.replay_triggered:v.replay_used].map(x=>cell(fmt(x))).join('')+'</tr>';
  }).join('');
  const a=s.assessment||{}, e=s.evaluation_progress||{}, internal=s.internal_evaluation||{}, initial=internal.report;
  const resolved=initial?initial.wins+initial.draws+initial.losses:0;
  const elo=resolved>0?internal.elo:null;
  $('elo-metrics').innerHTML=card('Elo interne provisoire · Gen5 vs Gen5 initiale',elo==null?'—':(elo>0?'+':'')+fmt(elo,1))+card('Gen5 initiale · référence fixe','0 Elo')+card('Lot évalué',internal.index==null?'—':fmt(internal.index+1))+(internal.version==null?'':card('Version évaluée',fmt(internal.version)));
  $('elo').textContent=internal.index==null?t(o.history_interval===0?'Aucune évaluation disponible.':'Première évaluation en attente.'):t('Valeur provisoire · {resolved} / {total} parties résolues contre le modèle initial.',{resolved:fmt(resolved),total:fmt(resolved+(initial?.unknown||0))})+' '+t(initial.unknown===0?'Panel complet ; estimation encore provisoire.':'Panel incomplet : trop de résultats manquants pour conclure à un progrès.')+' '+t('Gen5 initiale = 0 Elo. Cette comparaison mesure la progression face à sa propre version de départ, sans attendre Gen3.1.')+' '+[['initial',initial],...(a.references||[]).filter(([name])=>name!=='initial')].map(([name,r])=>t('{name} : {wins} V / {draws} N / {losses} D / {unknown} indéterminées',{name:name==='initial'?t('Référence initiale'):'Gen3.1',wins:fmt(r.wins),draws:fmt(r.draws),losses:fmt(r.losses),unknown:fmt(r.unknown)})).join(' ; ');
  $('evaluation-progress').textContent=o.history_interval===0?t('Évaluations séparées désactivées dans ce mode. La progression est suivie par les paliers Gen3.1.'):e.index==null?t('Évaluations espacées d’au moins {minutes} minutes, sous le plafond CPU historique.',{minutes:fmt(o.history_interval/60)}):t('{status} · {done} / {total} parties traitées, dont {resolved} résolues.',{status:t(e.complete?'Évaluation terminée':'Évaluation en cours'),done:fmt(e.processed),total:fmt(e.planned),resolved:fmt(e.resolved)});
  if(o.opponents?.length)$('evaluation-progress').textContent=t('Progression suivie séparément pour les cinq adversaires.');
  renderHourly(s);
  $('games').innerHTML=(s.games||[]).map(g=>'<tr>'+[g.id,g.collector_version,(g.lane==='Historical'&&g.opponent?'Gen5 ↔ '+g.opponent:(names[g.lane]||g.lane))+(g.reference_budget?' · MCTS-'+g.reference_budget:''),t(g.gen5_seat||'—'),resultLabel(g),(g.prefix_decisions?fmt(g.continuation_decisions)+' ('+fmt(g.prefix_decisions)+' → '+fmt(g.decisions)+')':fmt(g.decisions)),fmt(g.seconds,2)].map(x=>'<td>'+esc(x)+'</td>').join('')+'<td><a href="/examples/files/'+g.id+'.psr" download>'+t('Télécharger')+'</a> · <button onclick="copy('+g.id+')">'+t('Copier')+'</button></td></tr>').join('');
  $('error').textContent=[...(p.errors||[]),p.async_error,s.state.error].filter(Boolean).join(' · ');
}
const parisTime = timestamp => new Date(timestamp*1000).toLocaleTimeString(locale(), {timeZone:'Europe/Paris',hour:'2-digit',minute:'2-digit',hour12:false});
function hourLabel(timestamp, now, span=3600) {
  const date=new Date(timestamp*1000).toLocaleDateString(locale(), {timeZone:'Europe/Paris',day:'2-digit',month:'2-digit'});
  return date+' · '+parisTime(timestamp)+'–'+parisTime(timestamp+span)+(timestamp+span>now?' · '+t('En cours · partielle'):'');
}
function renderHourly(s) {
  const ensemble=!!s.options?.opponents?.length;
  const legacy=s.options?.legacy_replay||ensemble;
  const choices=[['Historical',t(ensemble?'Gen5 · toutes les références':'Gen5 contre Gen3.1')],
    ...(s.options?.opponents||[]).map(o=>['Gen'+o.generation,t('Gen5 contre {generation}',{generation:'Gen'+o.generation})]),
    ['Selfplay',t('Auto-jeu Gen5')],['Checkpoint',t('Gen5 · versions retenues')]];
  const lane=syncDashboardLane($('hourly-lane'),'gen5:'+s.campaign+':'+(ensemble?'ensemble':legacy?'legacy':'standard'),choices,ensemble?'Gen3.1':'Historical');
  $('history-heading').textContent=t(legacy?'Évolution par tranches de dix minutes':'Évolution heure par heure');
  $('history-description').textContent=t(legacy?'Campagne actuelle · Gen5 contre Gen3.1 · résultats séparés par budget MCTS.':'Tranches horaires fixes · heure de Paris · résultats enregistrés, reprises comprises. L’heure en cours est partielle.');
  if(ensemble)$('history-description').textContent=t('Campagne actuelle · résultats séparés par adversaire et budget MCTS.');
  const history=s.hourly_results||{}, observations=s.elo_timeline||{}, now=s.read_at_unix_seconds||Date.now()/1000;
  const cells=values=>values.map(v=>'<td>'+esc(v)+'</td>').join('');
  const bins=new Map((history.hours||[]).map(h=>[h.start,h]));
  const times=[...bins.keys(),...(observations.measurements||[]).map(r=>Math.floor(r.published/3600)*3600)];
  const hours=[];
  if(times.length)for(let h=Math.floor(now/3600)*3600;h>=Math.min(...times);h-=3600)hours.push(h);
  $('hourly-status').textContent=history.error?t('Historique indisponible : {error}',{error:history.error}):!history.ready?t('Chargement de l’historique…'):'';
  $('hourly-games').innerHTML=!history.ready?'':hours.map(h=>{
    const row=bins.get(h)?.lanes?.[lane]||{games:0,terminal:0,wins:0,losses:0,draws:0,fresh_used:0,ready:true};
    const wins=lane==='Selfplay'?'<td colspan="2">'+esc(t('Deux Gen5 · {count} parties décisives',{count:fmt(row.terminal-row.draws)}))+'</td>':cells([row.ready?fmt(row.wins):'—',row.ready?fmt(row.losses):'—']);
    return '<tr>'+cells([hourLabel(h,now),fmt(row.terminal)+' / '+fmt(row.games)])+wins+cells([fmt(row.draws),lane!=='Selfplay'&&row.ready&&row.terminal?fmt(100*row.wins/row.terminal,1)+' %':'—',fmt(row.fresh_used)])+'</tr>';
  }).join('')||(history.ready?'<tr><td colspan="7">'+esc(t('Aucune partie enregistrée.'))+'</td></tr>':'');
  const table=$('hourly-games').closest('table');
  table.querySelector('thead').innerHTML='<tr>'+['Heure',...(legacy?[...(ensemble?['Adversaire']:[]),'Budget MCTS']:[]),'Terminées / tentatives','Gen5 gagne','Adversaire gagne','Nulles','Taux de victoire','Positions neuves apprises'].map(label=>'<th>'+esc(t(label))+'</th>').join('')+'</tr>';
  if(legacy) {
    const history=s.replay_ten_minutes||{};
    $('hourly-status').textContent=history.error?t('Historique indisponible : {error}',{error:history.error}):!history.ready?t('Chargement de l’historique…'):'';
    $('hourly-games').innerHTML=gen5MatchRows(history,lane).map(row=>{
      const selfplay=row.lane==='Selfplay';
      const budget=row.budget==null?(selfplay?(s.options.budgets||[]).map(([b])=>b).join(' / '):'—'):'MCTS '+row.budget;
      const wins=selfplay?'<td colspan="2">'+esc(t('Deux Gen5 · {count} parties décisives',{count:fmt(row.terminal-row.draws)}))+'</td>':cells([row.seats_known?fmt(row.wins):'—',row.seats_known?fmt(row.losses):'—']);
      return '<tr>'+cells([hourLabel(row.start,now,600),...(ensemble?[row.generation||t(selfplay?'Auto-jeu Gen5':'Gen5 · versions retenues')]:[]),budget,fmt(row.terminal)+' / '+fmt(row.games)])+wins+cells([fmt(row.draws),!selfplay&&row.seats_known&&row.terminal?fmt(100*row.wins/row.terminal,1)+' %':'—',fmt(row.fresh_used)])+'</tr>';
    }).join('')||(history.ready?'<tr><td colspan="'+(ensemble?9:8)+'">'+esc(t('Aucune partie enregistrée.'))+'</td></tr>':'');
  }
  $('elo-timeline-status').textContent=observations.error?t('Historique indisponible : {error}',{error:observations.error}):'';
  const measurements=new Map();
  for(const r of observations.measurements||[]) {
    const key=Math.floor(r.published/3600)*3600+':'+r.reference;
    if(!measurements.has(key)||r.published>measurements.get(key).published)measurements.set(key,r);
  }
  $('hourly-elo').innerHTML=hours.flatMap(h=>['initial','gen3-1'].map(reference=>{
    const r=measurements.get(h+':'+reference), ref=reference==='initial'?t('Gen5 initiale'):'Gen3.1';
    return '<tr>'+cells([hourLabel(h,now),ref,...(r?[
      r.elo==null?t('Aucun résultat résolu'):(r.elo>0?'+':'')+fmt(r.elo,1),
      parisTime(r.published),fmt(r.version),[r.wins,r.draws,r.losses].map(v=>fmt(v)).join(' / '),fmt(r.unknown),fmt(r.simulations)+' / '+fmt(r.decision_limit)
    ]:[t('Aucune mesure'),'—','—','—','—','—'])])+'</tr>';
  })).join('');
}
async function refresh() {
  if(document.hidden)return;
  try {
    const response=await fetch('/api/status',{cache:'no-store'});
    if(!response.ok)throw new Error(response.status);
    const s=await response.json();
    if(latest?.campaign!==s.campaign)points=[];
    latest=s;render(s);
    if(typeof showingGen3==='function'&&showingGen3())return;
    if(s.gen5_terminal_per_second!=null&&!s.initializing&&s.controls?.running){points.push(s.gen5_terminal_per_second);points=points.slice(-120)}
    const max=Math.max(1,...points);
    $('spark').innerHTML='<polyline fill="none" stroke="#79c995" stroke-width="2" points="'+points.map((v,i)=>(i*1000/Math.max(1,points.length-1))+','+(100-90*v/max)).join(' ')+'"/>';
  } catch(e) { $('error').textContent=t('Lecture momentanément indisponible : {error}',{error:String(e)}); }
}
setLanguage(lang);
refresh();setInterval(refresh,5000);document.addEventListener('visibilitychange',refresh);
if(location.pathname==='/examples')$('examples').scrollIntoView();

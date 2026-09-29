// Independent lineage controls; status reads only bounded producer receipts.
let gen3Latest=null;
async function gen3Action(action,hours) {
  const status=document.getElementById('gen3-status');
  try {
    const response=await fetch('/api/gen3/control',{method:'POST',headers:{'Content-Type':'application/json','X-Paisho-Control':controlToken},body:JSON.stringify({action,hours})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||response.status);
    gen3Latest=data;await refreshGen3();
  }catch(e){status.textContent=String(e)}
}
function renderGen3() {
  if(!gen3Latest)return;
  const s=gen3Latest,p=s.progress||{},en=lang==='en';
  const gen5Busy=Boolean(latest?.controls?.running||latest?.controls?.busy);
  const active=['starting','running','pausing','stopping','fitting','evaluating'].includes(s.state);
  document.getElementById('gen3-title').textContent=(en?'Train Gen':'Entraîner Gen')+(s.generation||'3.2');
  for(const b of document.querySelectorAll('.gen3-duration'))b.disabled=active||!s.ready||gen5Busy;
  document.getElementById('gen3-pause').disabled=s.state!=='running';
  document.getElementById('gen3-stop').disabled=!['running','fitting','evaluating'].includes(s.state);
  document.getElementById('gen3-resume').disabled=active||gen5Busy||!(s.remaining_seconds>0);
  document.getElementById('gen3-next').disabled=active||!s.candidate||gen5Busy;
  document.getElementById('gen3-evaluate').disabled=active||!s.candidate||gen5Busy;
  const comparison=s.comparisons||[];
  document.getElementById('gen3-comparisons').innerHTML=comparison.length?'<table><thead><tr><th>MCTS</th><th>'+(en?'Wins / Draws / Losses':'Victoires / Nulles / Défaites')+'</th><th>'+(en?'Unknown':'Indéterminées')+'</th><th>'+(en?'Provisional internal Elo vs Gen3.1':'Elo interne provisoire contre Gen3.1')+'</th></tr></thead><tbody>'+comparison.map(r=>'<tr><td>'+Number(r.budget)+'</td><td>'+[r.wins,r.draws,r.losses].map(Number).join(' / ')+'</td><td>'+Number(r.unknown)+'</td><td>'+(r.conditional_internal_elo==null?'—':Number(r.conditional_internal_elo).toFixed(1))+'</td></tr>').join('')+'</tbody></table>':'';
  const recent=p.recent_games||[];
  document.getElementById('gen3-games').innerHTML=recent.length?'<table><thead><tr><th>MCTS</th><th>'+(en?'Opponent':'Adversaire')+'</th><th>'+(en?'Result':'Résultat')+'</th><th>'+(en?'Decisions':'Décisions')+'</th><th>PSR</th></tr></thead><tbody>'+recent.map(g=>'<tr><td>'+Number(g.budget)+'</td><td>'+(g.historical?(g.opponent?.startsWith('Gen3.')?(en?'Frozen '+g.opponent:g.opponent+' figée'):(en?'Historical heuristic':'Heuristique historique')):(en?'Self-play':'Auto-jeu'))+'</td><td>'+((g.termination!=='rules-terminal')?(en?'Interrupted':'Interrompue'):g.outcome==='Draw'?(en?'Draw':'Nulle'):!g.historical?(en?'Finished':'Terminée'):g.outcome==='Win('+g.candidate_seat+')'?(en?'Gen3 wins':'Gen3 gagne'):(en?'Opponent wins':'Adversaire gagne'))+'</td><td>'+Number(g.decisions)+'</td><td><a href="/gen3/games/'+Number(g.id)+'.psr">PSR '+Number(g.id)+'</a></td></tr>').join('')+'</tbody></table>':'';
  const names=en?{ready:'Ready',starting:'Starting',running:'Training',pausing:'Pausing',paused:'Paused',stopping:'Stopping',stopped:'Stopped',completed:'Completed',fitting:'Human value fit',evaluating:'Frozen comparison',failed:'Failed',unavailable:'Preparing'}:{ready:'Prête',starting:'Démarrage',running:'Entraînement',pausing:'Mise en pause',paused:'En pause',stopping:'Arrêt',stopped:'Arrêtée',completed:'Terminée',fitting:'Ajustement sur les parties humaines',evaluating:'Comparaison figée',failed:'Erreur',unavailable:'Préparation'};
  document.getElementById('gen3-status').textContent='Gen'+s.generation+' · '+(names[s.state]||s.state)+(p.completed!=null?' · '+p.terminal+'/'+p.completed+(en?' terminal games':' parties terminées')+' · '+p.updates+(en?' updates':' mises à jour'):'')+(gen5Busy&&!active?(en?' · available when Gen5 is paused or finished':' · disponible après la pause ou la fin de Gen5'):'')+(s.error?' · '+s.error:'');
}
async function refreshGen3(){
  try{const r=await fetch('/api/gen3/status',{cache:'no-store'});if(r.ok){gen3Latest=await r.json();renderGen3Protocol();renderGen3();if(latest)render(latest)}}catch(e){document.getElementById('gen3-status').textContent=String(e)}
}
refreshGen3();setInterval(()=>{if(!document.hidden)refreshGen3()},5000);
function renderGen3Protocol(){
 const en=lang==='en';
 const rows=en?[
 ['Value','Original Gen3.1 64 coefficients, unchanged at import; SGD 0.01, no L2, target 50% search + 50% terminal outcome.'],
 ['Games','Standard starts; 50% self-play, 50% historical heuristic; alternating seats. Fixed actor budgets repeating 512 / 256 / 128 / 64 / 32 / 512; each actor alternates opponents independently.'],
 ['Original cutoffs','32: 8 s · 64: 17.5 s · 128: 25 s · 256: 34 s · 512: 58 s; 2,048 decisions; at most 128 sampled roots per game.'],
 ['Search','Retained Gen3.1 UCT and candidate values. Added bounded learned action bias for ranking and tree selection; fades with visits. Not a replacement by Gen5 PUCT.'],
 ['Added policy','State trunk 128→32 and nonlinear residual action head with 16 units; starts neutral. Trained on actual search visits. Compact value remains independent.'],
 ['Memory','50,000 revalidated V2 games; all 1,375 human games kept, 291 held-out games excluded from retrieval. 20-turn segments; learned reader, frozen bank per campaign.'],
 ['RAM and CPU','65,536 replay positions as in Gen3.1, four rereads per fresh example; 32 GiB safety ceiling, ten private CPU game workers, ten actors, two archive workers; immutable weights and retained trees.'],
 ['Human fit','After a full run: original value fit, at most 10,000 epochs, patience 200, rate 0.1, seed 1. Epoch zero may win. Held-out data is used only for selection.'],
 ['Versions','Separate Gen3 and Gen5 weights and timers. Every session continues the same generation; moving to Gen3.3 requires an explicit joint decision. Compare before distributing: completion does not prove a strength gain.']
 ]:[
 ['Valeur','Les 64 coefficients de Gen3.1, inchangés à l’import ; SGD 0,01, sans L2, cible 50 % recherche + 50 % résultat terminal.'],
 ['Parties','Départs standards ; 50 % auto-jeu, 50 % adversaire heuristique historique ; places alternées. Budgets fixes par acteur : 512 / 256 / 128 / 64 / 32 / 512, répétés ; adversaires alternés localement.'],
 ['Coupures originales','32 : 8 s · 64 : 17,5 s · 128 : 25 s · 256 : 34 s · 512 : 58 s ; 2 048 décisions ; au plus 128 racines échantillonnées par partie.'],
 ['Recherche','UCT de Gen3.1, branches et valeurs des candidats conservées. Ajout d’un biais de politique borné pour le classement et la sélection, décroissant avec les visites. Pas de remplacement par le PUCT de Gen5.'],
 ['Politique ajoutée','Tronc de contexte 128→32 et tête résiduelle non linéaire de 16 neurones, initialement neutre. Apprentissage sur les visites de recherche. Valeur compacte indépendante.'],
 ['Mémoire','50 000 parties revalidées sous V2 ; les 1 375 parties humaines conservées, les 291 de validation exclues du rappel. Portions de 20 tours, lecteur appris, banque fixe pendant la campagne.'],
 ['RAM et CPU','65 536 positions de replay comme en Gen3.1, quatre relectures par exemple neuf ; plafond de 32 Gio, dix moteurs de partie sur dix threads CPU, dix acteurs, deux ouvriers d’archivage ; poids immuables et arbres conservés.'],
 ['Ajustement humain','Après une durée complète : ajustement de valeur original, au plus 10 000 époques, patience 200, taux 0,1, graine 1. L’époque zéro reste admissible. Validation réservée à la sélection.'],
 ['Versions','Poids et temps Gen3 séparés de Gen5. Toutes les sessions poursuivent Gen3.2 ; passer à Gen3.3 demande une décision explicite prise ensemble. Comparer avant diffusion : finir l’entraînement ne prouve pas un gain de force.']
 ];
 if(gen3Latest?.dashboard?.options?.historical_reference){
  rows[1]=[en?'Games':'Parties',en?'Standard starts; 90% self-play, 10% frozen Gen3.1. Each actor schedules one reference game in ten, alternating seats. Fixed actor budgets: 512 / 256 / 128 / 64 / 32 / 512.':'Départs standards ; 90 % auto-jeu, 10 % Gen3.1 figée. Chaque acteur programme une confrontation sur dix, avec alternance des places. Budgets fixes : 512 / 256 / 128 / 64 / 32 / 512.'];
  rows[2]=[en?'Limits':'Limites',en?'Self-play: 32: 8 s · 64: 17.5 s · 128: 25 s · 256: 34 s · 512: 58 s. Frozen Gen3.1: no individual wall timeout. Both: 2,048 decisions, repetitions and campaign deadline.':'Auto-jeu : 32 : 8 s · 64 : 17,5 s · 128 : 25 s · 256 : 34 s · 512 : 58 s. Gen3.1 figée : aucune coupure murale individuelle. Dans les deux cas : 2 048 décisions, répétitions et fin de campagne.'];
 }
 if(['3.4','3.5'].includes(gen3Latest?.generation)){
  const generation=gen3Latest.generation, references=[...new Set((gen3Latest?.dashboard?.options?.historical_pool||[]).map(e=>e.generation))].sort().join(', ');
  const historicalPercent=100/(gen3Latest?.dashboard?.options?.historical_every||10), selfplayPercent=100-historicalPercent;
  rows[0]=[en?'Value':'Valeur',en?'128 features: preserved parent coefficients plus 64 additional reserve, spatial and phase coordinates. SGD 0.01; unchanged 50% search / 50% outcome target.':'128 caractéristiques : coefficients parentaux conservés, plus 64 coordonnées de réserves, de répartition spatiale et de phase. SGD 0,01 ; cible inchangée 50 % recherche / 50 % résultat.'];
  rows[1]=[en?'Games':'Parties',en?`${selfplayPercent}% self-play / ${historicalPercent}% frozen ${references}, balanced across historical opportunities and seats. Budget-specific public experts.`:`${selfplayPercent} % auto-jeu / ${historicalPercent} % ${references} figées, réparties entre les confrontations historiques et les places. Experts publics propres à chaque budget.`];
  rows[2]=[en?'Limits':'Limites',en?'Self-play: 32: 8 s · 64: 17.5 s · 128: 25 s · 256: 34 s · 512: 58 s. Historical references: no individual time cap. All: 2,048 decisions, repetition detector and campaign deadline.':'Auto-jeu : 32 : 8 s · 64 : 17,5 s · 128 : 25 s · 256 : 34 s · 512 : 58 s. Références historiques : aucune coupure individuelle. Tous : 2 048 décisions, détecteur de répétitions et fin de campagne.'];
  rows[5]=[en?'Memory':'Mémoire',en?'50,000 original games preserved; 231,141 main sequences plus 231,211 bonus anchors. Learned recall at the root and internal nodes, including bonuses; held-out exclusions retained.':'50 000 parties originales conservées ; 231 141 séquences principales et 231 211 souvenirs de bonus. Rappel appris à la racine et dans les nœuds internes, bonus compris ; exclusions de validation conservées.'];
  rows[6]=[en?'RAM and CPU':'RAM et CPU',en?'48 GiB replay ceiling; 65,536 positions, four rereads; ten CPU actors and two archive workers. Unique frozen models and banks preloaded once, shared in RAM; private retained trees.':'Plafond du tampon : 48 Gio ; 65 536 positions, quatre relectures ; dix acteurs CPU et deux archiveurs. Modèles figés distincts et banques préchargés une fois, partagés en RAM ; arbres privés conservés.'];
  rows[7]=[en?'End of session':'Fin de session',en?'Durable 128-feature checkpoint retained. The legacy 64-input post-fit is not applied to this architecture. Compare the frozen candidate after training.':'Checkpoint durable à 128 caractéristiques conservé. L’ancien ajustement final à 64 entrées ne s’applique pas à cette architecture. Comparaison du candidat figé après entraînement.'];
  rows[8]=[en?'Versions':'Versions',en?`Gen${generation} continues across sessions. Earlier campaign states and public models are preserved separately.`:`Gen${generation} reste la même entre les sessions. Les états des campagnes et modèles publics antérieurs sont conservés séparément.`];
  if(generation==='3.5'){
   rows[0][1]+=en?' Added value residual 128→16→1 (2,081 parameters), initially neutral, trained and saved with the model.':' Résidu de valeur ajouté 128→16→1 (2 081 paramètres), initialement neutre, entraîné et sauvegardé avec le modèle.';
   rows[1][1]+=en?' Frozen Gen3.4 is the published approximately 22-minute checkpoint, also used as the starting parent.':' La Gen3.4 figée est le checkpoint publié après environ 22 minutes, également utilisé comme parent initial.';
   rows[3][1]+=en?' Corrected solver draw bounds propagate to selection, backup and learning targets.':' Les bornes de nulle corrigées du solveur sont conservées dans la sélection, la remontée et les cibles.';
  }
 }
 document.getElementById('gen3-protocol').innerHTML='<table><tbody>'+rows.map(([a,b])=>'<tr><th>'+a+'</th><td>'+b+'</td></tr>').join('')+'</tbody></table>';
 document.getElementById('gen3-next').textContent=en?'Next version':'Version suivante';
 document.getElementById('gen3-evaluate').textContent=en?'Compare with Gen3.1':'Comparer à Gen3.1';
}
renderGen3Protocol();document.getElementById('language').addEventListener('click',()=>{renderGen3Protocol();renderGen3()});

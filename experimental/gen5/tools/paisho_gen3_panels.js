// Reuse the main dashboard panels; keep training controls and lineages independent.
let dashboardView=new URLSearchParams(location.search).get('view')||localStorage.getItem('paisho-dashboard-view')||'auto';
let gen3PanelActive=false, gen3Plot=[];
const gen3Original=new Map();
const sharedSelectors=['main h1','#elo-explanation','#seats-explanation','#timeline-description'];
function showingGen3(){
  if(dashboardView==='gen5')return false;
  if(dashboardView==='gen3')return true;
  return !!gen3Latest?.campaign && (['starting','running','pausing','paused','stopping','fitting','evaluating'].includes(gen3Latest.state)||!latest?.controls?.running);
}
function selectDashboard(value){dashboardView=value;localStorage.setItem('paisho-dashboard-view',value);if(latest)render(latest)}
$('dashboard-view').value=dashboardView;
function gen3Save(element){if(element&&!gen3Original.has(element))gen3Original.set(element,element.innerHTML)}
function gen3Set(element,html){if(element){gen3Save(element);element.innerHTML=html}}
function restoreGen3Panels(){
  if(!gen3PanelActive)return;
  for(const [element,html] of gen3Original){element.innerHTML=html;const walker=document.createTreeWalker(element,NodeFilter.SHOW_TEXT);while(walker.nextNode()){const n=walker.currentNode;const text=n.textContent.trim();if(english[text])n.textContent=t(text)}}
  gen3Original.clear();gen3PanelActive=false;document.title='Pai Sho · Gen5';
  $('gen3-comparisons').hidden=false;$('gen3-games').hidden=false;
}
const gen3Text=(fr,en)=>lang==='en'?en:fr;
const gen3Cells=values=>values.map(x=>'<td>'+esc(x)+'</td>').join('');
function gen3Outcome(g){
  if(g.termination!=='rules-terminal')return gen3Text('Interrompue','Interrupted')+' · '+(g.termination||'');
  if(g.outcome==='Draw')return gen3Text('Partie nulle','Draw');
  if(!g.historical)return gen3Text('Auto-jeu terminé','Self-play finished');
  return g.outcome==='Win('+g.candidate_seat+')'?gen3Text('Gen3 gagne','Gen3 wins'):(g.opponent?.startsWith('Gen3.')?g.opponent:gen3Text('Heuristique','Heuristic'))+' '+gen3Text('gagne','wins');
}
async function copyGen3(id){
  try{const r=await fetch('/gen3/selected/'+id+'.psr');if(!r.ok)throw Error(r.status);await navigator.clipboard.writeText(await r.text());$('copy-status').textContent=gen3Text('PSR copié.','PSR copied.')}
  catch(e){$('copy-status').textContent=String(e)}
}
function renderGen3Panels(){
  if(!showingGen3()){restoreGen3Panels();return false}
  const s=gen3Latest,d=s?.dashboard;
  if(!d){$('main').innerHTML=card('Gen3',gen3Text('Lecture de la campagne…','Reading run…'));return true}
  gen3PanelActive=true;
  document.title='Pai Sho · Gen'+s.generation;
  if(!latest?.controls?.prepared_next?.protocol?.structural_repair)$('training-action-status').textContent=latest?.controls?.stopped?gen3Text('Gen5 arrêtée','Gen5 stopped'):'';
  for(const button of document.querySelectorAll('.train-duration,.legacy-duration'))button.disabled=['starting','running','pausing','stopping','fitting','evaluating'].includes(s.state);
  $('spark').setAttribute('aria-label','Gen'+s.generation+' · '+gen3Text('débit sur dix minutes','ten-minute throughput'));
  gen3Set($('spark').nextElementSibling,esc(gen3Text('Débit des compteurs natifs, sur dix minutes observées maximum ; moyenne de la session en attendant deux relevés.','Native counter throughput over at most ten observed minutes; session average until two readings are available.')));
  $('gen3-comparisons').hidden=true;$('gen3-games').hidden=true;
  const p=s.progress||{},o=d.options||{},generation='Gen'+s.generation,now=d.read_at;
  const name=k=>k==='Selfplay'?gen3Text('Auto-jeu ','Self-play ')+generation:generation+(k.startsWith('Gen3.')?gen3Text(' contre ',' vs ')+k+gen3Text(' figée',' frozen'):gen3Text(' contre MCTS heuristique',' vs heuristic MCTS'));
  gen3Set(document.querySelector('h1'),generation);
  $('dashboard-view').querySelector('[value="gen3"]').textContent=generation;
  const remaining=d.remaining_seconds||0,elapsed=p.elapsed_seconds||0;
  const states={running:gen3Text('En cours','Running'),paused:gen3Text('En pause','Paused'),stopped:gen3Text('Arrêtée','Stopped'),completed:gen3Text('Terminée','Completed'),fitting:gen3Text('Ajustement humain','Human fit'),evaluating:gen3Text('Évaluation','Evaluation'),starting:gen3Text('Démarrage','Starting')};
  const totalGames=Object.values(d.totals||{}).reduce((a,v)=>a+v.games,0);
  $('main').innerHTML=card('État',states[s.state]||s.state)+card(gen3Text('Version des poids publiée','Published weight version'),fmt(p.updates))+card('Parties archivées',fmt(totalGames))+card('Fin prévue',s.end&&s.state==='running'?new Date(s.end*1000).toLocaleTimeString(locale(),{hour:'2-digit',minute:'2-digit'}):'—');
  $('state').textContent=gen3Text('Même génération entre les sessions','Same generation across sessions')+' · '+gen3Text('Actualisation','Updated')+' '+fmt(d.progress_age,1)+' s · '+fmt(remaining/60,1)+' min';
  $('time').value=elapsed+remaining?100*elapsed/(elapsed+remaining):0;
  const totals=Object.values(d.totals||{}),fresh=totals.reduce((a,v)=>a+v.fresh,0),replay=totals.reduce((a,v)=>a+v.replay,0);
  $('learning').innerHTML=card(generation+' · '+gen3Text('parties / s','games / s'),d.ready?fmt(d.terminal_per_second,3):'—')+card(gen3Text('Fenêtre du débit','Throughput window'),fmt(d.window_seconds,1)+' s')+card('Checkpoint durable',fmt(d.checkpoint_updates))+card('Positions neuves apprises',fmt(fresh))+card('Relectures',fmt(replay))+card('Mémoire de positions',fmt(p.replay_positions))+card('RAM du tampon',fmt((p.replay_bytes||0)/2**30,2)+' GiB')+card(gen3Text('Réanalyses','Reanalyses'),'0')+card(gen3Text('Sondages mémoire','Memory queries'),gen3Text('Non mesurés','Not measured'))+card(gen3Text('Modèles historiques en RAM','Resident historical models'),fmt(p.historical_models_resident))+card(gen3Text('Plafond mémoire du tampon','Replay memory ceiling'),fmt((o.replay_max_bytes||0)/2**30)+' GiB')+card(gen3Text('Acteurs CPU','CPU actors'),fmt(o.actors))+card(gen3Text('Ouvriers d’archivage','Archive workers'),fmt(o.archive_workers));
  const historicalPercent=100/(o.historical_every||10), selfplayPercent=100-historicalPercent;
  const historicalNames=[...new Set((o.historical_pool||[]).map(e=>e.generation))].sort().join(', ');
  $('recipe').textContent=(o.historical_pool?.length?gen3Text(`UCT conservé · ${selfplayPercent} % auto-jeu / ${historicalPercent} % références figées ${historicalNames} · `,`Retained UCT · ${selfplayPercent}% self-play / ${historicalPercent}% frozen ${historicalNames} · `):o.historical_reference?gen3Text('UCT conservé · 90 % auto-jeu / 10 % Gen3.1 figée · ','Retained UCT · 90% self-play / 10% frozen Gen3.1 · '):gen3Text('UCT conservé · alternance auto-jeu / MCTS heuristique · ','Retained UCT · alternating self-play / heuristic MCTS · '))+[...new Set(o.budgets||[])].sort((a,b)=>a-b).join(' / ')+gen3Text(' simulations. Réanalyse non activée.',' simulations. Reanalysis is not enabled.');
  $('ladder-status').textContent='';
  const observations=d.evaluations||[],latestByBudget=new Map();
  for(const r of observations)if(!latestByBudget.has(r.budget)||r.published>latestByBudget.get(r.budget).published)latestByBudget.set(r.budget,r);
  const online=d.online_evaluations||[],onlineLatest=new Map();
  for(const r of online)if(!onlineLatest.has(r.reference+'|'+r.budget))onlineLatest.set(r.reference+'|'+r.budget,r);
  $('elo-metrics').innerHTML=card(gen3Text('Versions figées évaluées','Frozen versions evaluated'),fmt(d.evaluated_versions))+card(gen3Text('Lots d’évaluation figée','Frozen evaluation batches'),fmt(observations.length))+[...latestByBudget].sort((a,b)=>a[0]-b[0]).map(([budget,r])=>card(generation+' / Gen3.1 · MCTS '+budget,r.conditional_internal_elo==null?'—':fmt(r.conditional_internal_elo,1)+' Elo')).join('');
  $('elo-metrics').innerHTML += [...onlineLatest].sort((a,b)=>a[1].budget-b[1].budget).map(([,r])=>card(gen3Text('Entraînement · Δ Elo provisoire','Training · provisional Δ Elo')+' · '+r.reference+' · MCTS '+r.budget+' · '+fmt(r.terminal)+' '+gen3Text('terminées','finished'),r.conditional_internal_elo==null?'—':fmt(r.conditional_internal_elo,1))).join('');
  $('elo').textContent=online.length?gen3Text('Δ Elo contre chaque référence figée, ancre 0, par budget et tranche d’une heure. Les poids évoluent dans chaque tranche : mesure descriptive conditionnelle aux parties terminées, pas une évaluation du dernier checkpoint.','Δ Elo against each frozen reference, anchor 0, by budget and hourly bin. Weights change within each bin: descriptive measurement conditional on finished games, not an evaluation of the latest checkpoint.'):observations.length?gen3Text('Elo conditionnel par budget contre Gen3.1 figée ; les interruptions ne sont pas des nulles.','Conditional Elo by budget against frozen Gen3.1; interruptions are not draws.'):gen3Text('Aucune évaluation figée disponible. Les résultats d’entraînement apparaîtront séparément pour chaque référence, sans les confondre avec un test du dernier modèle.','No frozen evaluation available. Training results will appear separately for each reference; they are not a test of the latest model.');
  $('evaluation-progress').textContent=s.state==='evaluating'?gen3Text('Comparaison en cours : ','Comparison in progress: ')+fmt(d.evaluation_progress?.processed)+' / '+fmt(d.evaluation_progress?.planned)+gen3Text(' parties traitées.',' games processed.'):gen3Text('La commande « Comparer à Gen3.1 » lance les évaluations après la session. Elles ne se déclenchent pas automatiquement.','“Compare with Gen3.1” runs evaluations after the session. They do not start automatically.');
  gen3Set($('elo').parentElement.querySelector('p.note:last-child'),esc(gen3Text('Chaque mesure conserve le modèle testé, la référence et son budget. Aucune valeur n’est inventée avant une évaluation.','Each measurement retains the tested model, reference and budget. No rating is invented before an evaluation.')));
  const hourly=period==='hour',displayed=hourly?d.last_hour:d.totals;
  $('period-total').setAttribute('aria-pressed',String(!hourly));$('period-hour').setAttribute('aria-pressed',String(hourly));
  $('period-note').textContent=gen3Text(hourly?'Parties publiées durant la dernière heure.':'Totaux de cette génération, reprises comprises.',hourly?'Games published during the last hour.':'Generation totals, including resumed sessions.')+(!d.ready?' · '+gen3Text('Import de l’historique en cours.','Loading history.'):'');
  gen3Set($('lanes').closest('table').querySelector('thead'),'<tr>'+['File','Terminées / tentatives',generation+' '+gen3Text('gagne','wins'),gen3Text('Adversaire gagne','Opponent wins'),'Nulles','Décisions','Positions neuves apprises','Relectures'].map(x=>'<th>'+esc(t(x))+'</th>').join('')+'</tr>');
  $('lanes').innerHTML=Object.entries(displayed||{}).map(([lane,r])=>'<tr>'+gen3Cells([name(lane),fmt(r.terminal)+' / '+fmt(r.games)])+(lane==='Selfplay'?'<td colspan="2">'+esc(gen3Text('Deux modèles identiques · ','Two identical models · ')+fmt(r.terminal-r.draws)+gen3Text(' décisives',' decisive'))+'</td>':gen3Cells([fmt(r.wins),fmt(r.losses)]))+gen3Cells([fmt(r.draws),fmt(r.decisions),fmt(r.fresh),fmt(r.replay)])+'</tr>').join('');
  gen3Set($('lanes').closest('section').querySelector('p:last-child'),esc(gen3Text('Les places alternent. Les victoires sont attribuées à Gen3 ou à l’adversaire ; l’auto-jeu ne mesure pas un gain de force.','Seats alternate. Wins belong to Gen3 or its opponent; self-play does not measure strength gains.')));
  $('history-heading').textContent=gen3Text('Évolution heure par heure','Hourly progress');
  $('history-description').textContent=gen3Text('Parties publiées, séparées par adversaire et budget MCTS.','Published games, separated by opponent and MCTS budget.');
  const references=[...new Set(['Gen3.1','Gen3.2','Gen3.3',...(o.historical_pool||[]).map(e=>e.generation)])];
  const choices=[...(o.historical_pool?.length?[]:[['Historical',name('Historical')]]),['Selfplay',name('Selfplay')],...references.map(k=>[k,name(k)])];
  const lane=syncDashboardLane($('hourly-lane'),'gen3:'+s.generation+':'+(s.campaign||''),choices,o.historical_pool?.length?'Gen3.3':o.historical_reference?'Gen3.1':'Historical');
  $('hourly-status').textContent=d.ready?'':t('Chargement de l’historique…');
  gen3Set($('hourly-games').closest('table').querySelector('thead'),'<tr>'+['Heure','MCTS','Terminées / tentatives',generation+' '+gen3Text('gagne','wins'),gen3Text('Adversaire gagne','Opponent wins'),'Nulles','Taux de victoire','Positions neuves apprises'].map(x=>'<th>'+esc(t(x))+'</th>').join('')+'</tr>');
  $('hourly-games').innerHTML=(d.bins||[]).filter(r=>r.lane===lane).map(r=>'<tr>'+gen3Cells([hourLabel(r.start,now,d.bin_seconds||3600),r.budget,fmt(r.terminal)+' / '+fmt(r.games)])+(lane==='Selfplay'?'<td colspan="2">'+esc(fmt(r.terminal-r.draws)+gen3Text(' décisives',' decisive'))+'</td>':gen3Cells([fmt(r.wins),fmt(r.losses)]))+gen3Cells([fmt(r.draws),lane!=='Selfplay'&&r.terminal?fmt(100*r.wins/r.terminal,1)+' %':'—',fmt(r.fresh)])+'</tr>').join('')||'<tr><td colspan="8">'+esc(gen3Text('Aucune partie dans cette file.','No game in this lane.'))+'</td></tr>';
  const onlineRows=online.map(r=>'<tr>'+gen3Cells([hourLabel(r.start,now,d.bin_seconds||3600),r.reference+' · MCTS '+r.budget,fmt(r.conditional_internal_elo,1),parisTime(r.published),r.version_min+' → '+r.version_max,[r.wins,r.draws,r.losses].join(' / '),r.unknown,gen3Text('Entraînement · provisoire','Training · provisional')])+'</tr>').join('');
  const byHour=new Map();for(const r of observations){const key=Math.floor(r.published/3600)*3600+':'+r.budget;if(!byHour.has(key)||r.published>byHour.get(key).published)byHour.set(key,r)}
  $('elo-timeline-status').textContent='';
  gen3Set($('hourly-elo').closest('section').querySelector('h2'),esc(gen3Text('Historique Elo · entraînement et évaluations','Elo history · training and evaluations')));
  gen3Set($('hourly-elo').closest('table').querySelector('thead'),'<tr>'+[gen3Text('Période','Period'),gen3Text('Référence','Reference'),'Δ Elo',gen3Text('Mesuré à','Measured at'),gen3Text('Versions jouées','Played versions'),'V / N / D',gen3Text('Indéterminées','Unknown'),gen3Text('Type / limites','Type / limits')].map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr>');
  gen3Set($('hourly-elo').closest('section').querySelector('p.note'),esc(gen3Text('Entraînement : tranches d’une heure et plages des versions jouées. Évaluations figées : dernière mesure de chaque heure. Les interruptions restent indéterminées.','Training: hourly bins and played version ranges. Frozen evaluations: last measurement per hour. Interruptions remain unknown.')));
  $('hourly-elo').innerHTML=onlineRows+[...byHour.values()].sort((a,b)=>b.published-a.published||a.budget-b.budget).map(r=>'<tr>'+gen3Cells([hourLabel(Math.floor(r.published/3600)*3600,now),r.reference+' · MCTS '+r.budget,fmt(r.conditional_internal_elo,1),parisTime(r.published),(r.model_sha256||'').slice(0,12),[r.wins,r.draws,r.losses].join(' / '),r.unknown,r.budget+' / 2048'])+'</tr>').join('')||'<tr><td colspan="8">'+esc(t('Aucune mesure'))+'</td></tr>';
  gen3Set($('games').closest('table').querySelector('thead'),'<tr>'+['Partie','Modèle joué','File',gen3Text('Place de Gen3','Gen3 seat'),'Résultat','Décisions','Secondes','PSR'].map(x=>'<th>'+esc(t(x))+'</th>').join('')+'</tr>');
  $('games').innerHTML=(d.games||[]).map(g=>'<tr>'+gen3Cells([g.id,g.collector_version,name(g.historical?(g.opponent?.startsWith('Gen3.')?g.opponent:'Historical'):'Selfplay')+' · MCTS '+g.budget,g.candidate_seat,gen3Outcome(g),g.decisions,fmt(g.seconds,2)])+'<td><a download href="/gen3/selected/'+g.download_id+'.psr">'+t('Télécharger')+'</a> · <button onclick="copyGen3(\''+g.download_id+'\')">'+t('Copier')+'</button></td></tr>').join('');
  $('error').textContent=[...(d.errors||[]),s.error].filter(Boolean).join(' · ');
  if(gen3Plot.at(-1)?.time!==now&&d.ready){gen3Plot.push({time:now,value:d.terminal_per_second});gen3Plot=gen3Plot.filter(x=>now-x.time<=600)}
  const max=Math.max(1,...gen3Plot.map(x=>x.value));$('spark').innerHTML='<polyline fill="none" stroke="#79c995" stroke-width="2" points="'+gen3Plot.map((x,i)=>(i*1000/Math.max(1,gen3Plot.length-1))+','+(100-90*x.value/max)).join(' ')+'"/>';
  return true;
}

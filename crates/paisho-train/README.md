# Acteurs et learner MPSGraph durables

## Apprentissage compact CPU courant

L'ancienne campagne PPO est abandonnée, sauvegardes conservées. Le suivi courant
est dans `docs/tracking/COMPACT_MCTS_ROADMAP.md`. `paisho-compact` entraîne une
petite fonction de valeur qui guide le MCTS, avec les mêmes règles pour les
budgets32/64/128/256/512. Le budget de simulations est indépendant des64 poids.

```sh
python3 tools/benchmark_with_training_paused.py --offline-paused \
  --config training-runs/control/paisho-control.json -- \
  target/release/paisho-compact selfplay --model parent.json --output nouveau-lot \
  --simulations 64 --workers 8 --games 100000 --seconds 120 \
  --decision-limit 2048 --samples 128
```

Les limites de temps sont souples. Seules les parties réglementairement terminées
produisent des cibles ; toutes les parties et les versions utilisées restent
archivées. `--learn false` mesure un modèle figé. Le profil fixé par
COMPACT_MCTS_V5 fournit les coupures par défaut : 32 = 8 s, 64 = 17,5 s,
128 = 25 s, 256 = 34 s, 512 = 58 s. Nouvelle mesure après génération différée,
sur demande explicite ; aucune recalibration automatique.
`--game-seconds` peut surcharger explicitement ce réglage ; le plafond global
reste prioritaire lorsqu'il expire avant. Le plan conserve la valeur appliquée.
Ces coupures ne s'appliquent pas à `compare`.

`compare --candidate modele.json --output nouveau-test --simulations 64
--reference-simulations 512` confronte des poids figés à l'ancienne heuristique512.
Sans `--reference-simulations`, les deux budgets sont égaux. Les sièges sont
inversés ; les temps effectifs et paires incomplètes sont conservés. Un adversaire
commun permet le suivi Elo ; seul un budget égal isole le gain à simulations
égales, et une mesure à temps comparable reste nécessaire avant adoption.

Les commandes PPO ci-dessous décrivent les capacités historiques et ne sont pas
une instruction de relancer la campagne abandonnée.

## Boucle persistante de la branche d'optimisation

`paisho-live` conserve le learner, Adam, le pool Rayon et les services acteurs
entre les cycles. Les poids seuls passent en RAM via PSW1/PSW2, après la fin
de toutes les parties du cycle précédent. Les nouveaux replays portent
l'empreinte de ces poids et les valeurs de comportement nécessaires à PPO.

```sh
target/release/paisho-live --campaign-dir /chemin/nouvelle-campagne \
  --initial-checkpoint /chemin/parent.psckpt --target-generation 100 \
  --workers 20 --actors 80 --actor-games 512 --actor-max-attempts 2048 \
  --actor-decision-limit 256 --steps-per-generation 256 \
  --durable-every 5 --promotion-every 10 --evaluation-every 10
```

La cible est un numéro de génération absolu. La même commande reprend depuis
le dernier `blocks/block-*.json` publié après le checkpoint vérifié. Les cycles
ultérieurs en RAM sont recalculés ; leurs replays restent dans leur ancienne
tentative, sans être comptés deux fois. Un arrêt propre à la cible publie aussi
un checkpoint, même avant le prochain multiple de cinq. Les évaluations dues
sont reprises avant de repartir. `--checkpoint-every` est un paramètre du
learner historique et ne pilote pas cette boucle : utiliser `--durable-every`.

`--actor-round-size 512` permet de tester des rondes plus grandes sans modifier
le plan historique. C'est un réglage d'ordonnancement, pas le nombre de workers
ni de parties retenues. La valeur effectivement utilisée et le fingerprint
du code sont écrits dans `attempts/.../runtime.json`. Sans ce réglage, `--actors`
reste utilisé. Les paires, graines par partie et règles de sélection restent
inchangées. Les premiers comparatifs sont dans
`docs/research/PPO_EXECUTABLE_AND_BROKER_WAIT_2026-09-06.md`.

`live-status.json` est une vue légère, non un point de reprise. Pour le
contrôleur local, `progress_directory` désigne la racine de cette campagne et
`target_training_step` reste `null` : la cible est générationnelle, et un retour
au champion peut restaurer un pas Adam plus ancien. Ne pas réutiliser le
répertoire d'une campagne historique pour ce nouveau format.

L'amorçage facultatif se prépare avec `tools/paisho_teacher_bootstrap.py plan`
(checkpoint et anciens snapshots explicites), puis se lance avec
`tools/paisho_live_launch.py --bootstrap-campaign DOSSIER --` suivi des options
de `paisho-live`. Ce lanceur fournit lui-même le checkpoint et l'adversaire.
Le professeur MCTS-8 est omis/retiré dès la domination indépendante de Random ;
sinon son budget borné se termine sans prétendre à une victoire, et PPO pur
continue contre Random. Aucun corpus du professeur n'entre dans PPO.

Tests matériels opt-in : `tools/smoke_live_campaign.py`, à exécuter avec
`tools/benchmark_with_training_paused.py`. Son fixture importe G47 ;
`--full-load` mesure deux cycles complets, le mode normal vérifie une
interruption après G51, la reprise depuis G50 et la conservation des replays.

## Boucle historique

Pour espacer les mesures, `paisho-generations --promotion-every 10
--evaluation-every 10` les exécute aux générations multiples de dix. Les deux
options valent 1 par défaut pour conserver le comportement historique.
Le calendrier est inscrit dans le plan de chaque génération et survit à une
reprise avec d'autres options. Entre les mesures, le candidat devient le parent
d'entraînement, le champion reste inchangé et aucun Elo n'est inventé.
`--checkpoint-every` compte toujours des pas Adam : un checkpoint final est
encore publié à chaque génération pour transmettre les poids au cycle suivant.

`paisho-train` consomme un snapshot immuable de `paisho-replay`, exécute des
lots Adam supervisés ou PPO terminal dans le service MPSGraph et publie des
checkpoints reprenables. La cible donnée en ligne de commande est une étape
absolue : relancer la même commande après un arrêt reprend le dernier jalon et
ne dépasse pas la cible.

Chaque jalon comporte deux fichiers immuables :

1. le checkpoint MPSGraph, qui contient poids, Adam et progression ;
2. un manifeste `PAISHO-LEARNER-COMMIT 3`, écrit seulement après vérification
   du checkpoint et liant son SHA-256 au snapshot, au curseur et à la
   configuration exacte.

Pour l'objectif PPO terminal, chaque checkpoint possède aussi un petit segment
`*.ppo-metrics.json`. Il conserve, lot par lot, les pertes, l'entropie,
l'avantage moyen, le ratio d'importance moyen et son écart quadratique à 1. Le
segment est lié au nom et au SHA-256 du checkpoint et publié avant le manifeste
du learner : si cette publication échoue, le checkpoint reste orphelin et ne
fait pas avancer la reprise. Ces métriques sont diagnostiques ; elles ne
deviennent jamais un critère de victoire à la place des résultats de parties.

Le lecteur accepte toujours les manifests V1 et V2 déjà publiés. Le V2 avait
ajouté le pas global de départ de la génération ; le V3 ajoute l'identité de
l'objectif, du producteur de comportement, de l'acteur et des paramètres PPO.

Après le premier checkpoint vérifié, chaque répertoire de génération contient
aussi un petit `origin.psorigin` V2. Il lie une fois pour toutes l'identité de la
génération au SHA-256, au numéro de génération et au pas global de son
checkpoint parent. Le curseur de replay est local à cette génération ; le pas
Adam demeure global afin de préserver sa correction de biais et les moments
restaurés. Un échec antérieur à ce premier checkpoint ne scelle rien : les
données sources restent intactes et le même répertoire peut être relancé avec
une entrée corrigée.

Une interruption avant le manifeste laisse un checkpoint orphelin. Celui-ci
n'avance pas la progression : le prochain lancement repart du manifeste
précédent, rejoue les lots et publie une nouvelle tentative sous un autre nom.
Cette distinction est nécessaire parce que deux exécutions Metal du même pas
peuvent être numériquement valides sans produire des octets identiques. Une
corruption du dernier jalon engagé provoque une erreur explicite ; elle n'est
pas masquée par un retour silencieux à un modèle plus ancien.

```sh
cargo run --release -p paisho-train -- \
  --snapshot /chemin/replay/snapshot.psrsnap \
  --replay-dir /chemin/replay \
  --run-dir /chemin/entrainement/generation-0000 \
  --target-step 1000 \
  --checkpoint-every 100 \
  --preset pure --batch 64 --actions 1024 --level 1
```

Cette commande conserve le mode supervisé historique. La voie pure principale
emploie le résultat terminal avec PPO :

```sh
cargo run --release -p paisho-train -- \
  --snapshot /chemin/replay/snapshot.psrsnap \
  --replay-dir /chemin/replay \
  --run-dir /chemin/entrainement/generation-0000 \
  --objective terminal-ppo \
  --behavior-producer EMPREINTE_SHA256_DE_L_ACTEUR \
  --policy-temperature 1.0 --uniform-mix 0.05 \
  --ppo-clip 0.2 --ppo-value-weight 0.5 --ppo-entropy-weight 0.01 \
  --target-step 1000 --checkpoint-every 100 \
  --preset pure --batch 64 --actions 1024 --level 1
```

Le dataset PPO ne retient que les politiques `Behavior` de ce producteur. Les
nouveaux replays enregistrent aussi le baseline `P(win)-P(loss)` déjà calculé
par l'acteur au moment du coup : le learner le précharge en RAM et ne lance
alors que le processus candidat muté. Un corpus V1 ancien reste compatible ;
si une valeur manque, un second processus acteur immuable recalcule toutes les
valeurs du lot. Pour une génération enfant, `--actor-checkpoint` continue de
désigner le même checkpoint que `--initial-checkpoint` afin d'en lier et d'en
vérifier l'identité, même lorsque son exécution supplémentaire n'est plus
nécessaire. À la reprise, le candidat repart toujours du dernier commit.

La composition d'un corpus peut être relue indépendamment avant de modifier un
réglage :

```sh
cargo run --release -p paisho-train --bin paisho-replay-stats -- \
  --snapshot /chemin/replay/snapshot.psrsnap \
  --replay-dir /chemin/replay \
  --behavior-producer EMPREINTE_SHA256_DE_L_ACTEUR
```

Cette commande vérifie et rejoue les shards, puis rapporte séparément parties
et décisions gagnantes, nulles et perdantes. Elle ne modifie aucun artefact.

La concentration d'un checkpoint peut être mesurée sur les mêmes positions,
en séparant les décisions à un seul coup légal et sans produire de partie :

```sh
cargo run --release -p paisho-train --bin paisho-policy-stats -- \
  --snapshot /chemin/replay/snapshot.psrsnap \
  --replay-dir /chemin/replay \
  --behavior-producer EMPREINTE_SHA256_DE_L_ACTEUR \
  --checkpoint /chemin/modele.psckpt \
  --examples 4096 --profiles 1:0.05,3:0.20 \
  --workers 10 --classes 64:8,128:4,1024:4
```

Le GPU calcule les politiques, les workers CPU alimentent les courtiers et les
profils sont appliqués aux mêmes sorties. L'outil rapporte notamment entropie
normalisée, nombre effectif de choix, masse maximale et `Σp²`, cette dernière
étant la probabilité moyenne attendue d'un coup échantillonné sous la même
distribution. Il s'agit d'un diagnostic en lecture seule, jamais d'un critère
de promotion.

La première exécution d'une génération enfant reçoit le checkpoint de son
parent. Elle peut adopter un nouveau snapshot et un nouveau taux, mais son
premier lot commence nécessairement à l'index local zéro :

```sh
cargo run --release -p paisho-train -- \
  --snapshot /chemin/replay/generation-0001.psrsnap \
  --replay-dir /chemin/replay \
  --run-dir /chemin/entrainement/generation-0001 \
  --initial-checkpoint /chemin/entrainement/generation-0000/modele.psckpt \
  --objective terminal-ppo --behavior-producer EMPREINTE_SHA256_DE_L_ACTEUR \
  --actor-checkpoint /chemin/entrainement/generation-0000/modele.psckpt \
  --generation 1 --target-step 2000 --checkpoint-every 100 \
  --preset pure --batch 64 --actions 1024 --level 1
```

Après le premier commit enfant, une reprise supervisée ordinaire n'a plus besoin
du chemin du parent : `origin.psorigin` conserve sa provenance et le checkpoint
local porte le nouvel état exact. En PPO, `--actor-checkpoint` reste l'identité
du producteur gelé ; le fichier n'est rouvert pour l'inférence que lorsqu'un
corpus compatible ancien ne contient pas ses valeurs WDL enregistrées.

Le banc de panne local crée volontairement un checkpoint sans manifeste,
vérifie que le learner le laisse orphelin, puis contrôle le passage réel à une
génération enfant et sa reprise locale :

```sh
cargo run --release -p paisho-train --example durable_learner_smoke -- \
  apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-service pure 8
```

Le smoke dédié au PPO vérifie aussi une reprise initiale et enfant avec les
décisions terminales des deux camps :

```sh
cargo run --release -p paisho-train --example terminal_ppo_learner_smoke -- \
  apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-service
```

## Professeur MCTS-8 expérimental hors ligne

`paisho-teacher-relabel` prépare un corpus supervisé pour l'amorçage décrit dans
[`CURRICULUM_V4.md`](../../docs/authority/CURRICULUM_V4.md). Il sélectionne sans
remplacement au plus `--positions` décisions `Behavior` de replays scellés,
selon un classement SHA-256 déterministe lié à `--seed`. Les shards identiques
fournis plusieurs fois sont dédupliqués. MCTS-8 analyse ces positions comme
professeur de politique : ses visites à la racine, normalisées à température 1,
deviennent des cibles `MctsVisit` sur les coups légaux. Les trajectoires et leurs
résultats terminaux sont conservés ; ces résultats restent les seules cibles
de valeur, dans la perspective du joueur au trait.

```sh
cargo run --release -p paisho-train --bin paisho-teacher-relabel -- \
  --source /chemin/generation-0001/replay \
  --source /chemin/generation-0002/replay/snapshot.psrsnap \
  --positions 1000 --seed 42 --workers 8 \
  --output /chemin/nouveau-corpus-teacher
```

Chaque `--source` désigne un snapshot ou un répertoire contenant
`snapshot.psrsnap` ; les shards doivent être à côté du snapshot.
`--workers N` doit être positif et vaut par défaut `available_parallelism`.
Les cibles indépendantes sont calculées dans un pool Rayon commun avec des
graines par décision. L'ordre de publication reste stable : à sources, graine
et binaire identiques, changer le nombre de workers conserve les octets des
shards et du snapshot. Seul le compteur `cpu_threads` de la provenance change.

La sortie doit être un nouveau répertoire dont le parent existe. Elle contient
les shards `teacher-*.psrshard`, `provenance.json` (sources, positions, graines,
configuration MCTS et empreintes des producteurs) et `snapshot.psrsnap`, publié
en dernier. Une interruption peut laisser un répertoire incomplet ; la reprise
se fait vers une nouvelle sortie. Le corpus contient uniquement les cibles du
professeur et se lit avec le learner supervisé ; il n'est pas un corpus PPO
`Behavior` et aucun mélange avec les anciennes politiques n'est appliqué.

Cet outil reste expérimental et hors ligne : il ne lance ni partie, ni learner,
ni service MPSGraph et ne modifie pas la campagne active. Il n'est pas encore
raccordé automatiquement à `paisho-generations`. Le choix d'une nouvelle
campagne, le mélange éventuel et l'arrêt de l'amorçage après le critère contre
`RandomAgent` restent à orchestrer selon la V4. Le réseau conserve une
inférence directe sans MCTS.

## Producteurs de parties

`paisho-actors` fait jouer le réseau pur directement, sans MCTS, et peut lui
opposer le même réseau, le bot aléatoire, `SiteBotV1` ou un MCTS de budget
choisi. Ses acteurs Rust occupent un pool Rayon tandis que leurs inférences
sont regroupées dans les classes fixes du service MPSGraph.

```sh
cargo run --release -p paisho-train --bin paisho-actors -- \
  --output-dir /chemin/acteurs --target-games 100 --max-attempts 400 \
  --opponent mcts:128 --checkpoint /chemin/modele.psckpt \
  --workers 10 --actors 80 --classes 64:8,128:4,1024:4 --wide-lanes 1 \
  --decision-limit 2048 --selection sample \
  --temperature 1.0 --uniform-mix 0.05 \
  --start-horizon 64
```

`--wide-lanes N` conserve ces tailles de lots mais crée `N` processus et files
MPSGraph indépendants pour la classe de plus grande capacité. Sur deux
balayages grandeur réelle de 512 parties, 2 files restent à égalité avec 1,
puis 3 à 5 fragmentent les lots et ralentissent la collecte. La valeur par
défaut reste donc 1 ; l'option sert aux mesures sur d'autres ordonnancements ou
machines.

Contre un adversaire distinct, deux manches de même setup sont toujours
produites ensemble avec les sièges inversés. Si l'une est interrompue, aucune
des deux n'entre dans le replay. Une campagne incomplète publie tout de même
les paires valides avant de rendre une erreur. Chaque exécution devient visible
en une seule opération sous `actor-run-*` avec shard, snapshot, télémétrie par
siège, description des départs, descriptions d'agents liées aux empreintes du
code, du binaire, du service et du checkpoint, puis manifeste SHA-256.

La sélection `sample`, employée par défaut pour produire les données PPO,
conserve la température et le mélange exploratoire dans l'identité de l'agent.
La sélection `argmax` joue au contraire le réseau sans exploration pour les
défis de curriculum. Contre un adversaire distinct, l'archive V4 inscrit alors
les victoires, nuls, défaites, cinq issues des paires et le test exact apparié.
Ces résultats restent explicitement `curriculum-terminal-non-elo` lorsqu'ils
proviennent de départs avancés : ils mesurent un progrès local, pas une force
comparable au classement du site.

`--start-horizon N` active les débuts avancés neutres. Une trajectoire complète
est produite par choix uniforme parmi tous les coups légaux, puis le préfixe
s'arrête à la frontière de tour principal dont la continuation source est la
plus proche de `N` décisions. Chaque paire à sièges inversés partage exactement
le même préfixe. La continuation source sert seulement à mesurer cette distance
et n'est jamais montrée au réseau ; les coups du préfixe sont rejoués pour
reconstruire l'état mais exclus des exemples PPO. Le résultat terminal de la
nouvelle partie reste l'unique récompense. `--start-seed`,
`--start-source-limit` et `--start-source-attempts` règlent la génération et
sont consignés avec la provenance par partie dans `starts.tsv`.

Un arrêt brutal au milieu d'un lot peut encore imposer de rejouer ce lot.

## Promotion entre checkpoints

`paisho-promote` oppose deux checkpoints de la même architecture, sans MCTS.
Il utilise un courtier MPSGraph multi-capacités distinct par jeu de poids,
tandis que les parties occupent le pool CPU. Une observation statistique est
toujours une paire au même setup avec sièges inversés. La politique est argmax
si aucune température n'est indiquée ; le mode échantillonné reproductible est
préférable au début du curriculum, lorsque deux argmax peuvent prolonger la
partie indéfiniment.

```sh
cargo run --release -p paisho-train --bin paisho-promote -- \
  --output-dir /chemin/promotions/generation-0001 \
  --candidate-checkpoint /chemin/candidat.psckpt \
  --champion-checkpoint /chemin/champion.psckpt \
  --max-eligible-pairs 400 --max-attempted-pairs 800 \
  --pairs-per-batch 10 --workers 10 \
  --start-horizon 64 --start-seed 20260905 \
  --sampling-temperature 1 --sampling-uniform-mix 0.05 \
  --elo0 0 --elo1 10 --alpha 0.05 --beta 0.05 \
  --preset pure --classes 64:8,128:4,1024:4 --level 1
```

Chaque lot devient visible par renommage atomique avec ses journaux et son
manifeste. Une relance avec la même identité repart du lot suivant. Si une
borne SPRT est atteinte, le résultat est `PromoteCandidate` ou
`RejectCandidate`; si un budget est épuisé avant une borne, il reste
explicitement indéterminé. Une partie interrompue exclut sa paire entière.
`--start-horizon` peut faire repartir les deux manches d'une paire du même
préfixe neutre, produit indépendamment des deux checkpoints et conservé dans
le journal complet. Cette promotion accélérée sélectionne un champion de
curriculum ; elle n'est ni une partie standard ni une mesure Elo. Sans cette
option, le duel commence selon les règles standard comme auparavant.
La politique d'évaluation est scellée elle aussi. L'échantillonnage reste une
politique du réseau pur, pas une recherche ; ses graines sont dérivées de la
paire, du siège et de la version des poids. Le GSPRT mesure donc la performance
attendue de ce protocole stochastique reproductible.

## Évaluation Elo contre les adversaires du curriculum

`paisho-evaluate` est séparé des producteurs de replay. Il oppose un checkpoint
figé à `random`, `site` ou `mcts:N` avec `N ≤ 512`, sur les mêmes paires à
sièges inversés et, facultativement, les mêmes départs neutres que les
promotions. Ses parties ne deviennent jamais des exemples PPO.

```sh
cargo run --release -p paisho-train --bin paisho-evaluate -- \
  --output-dir /chemin/evaluation \
  --candidate-checkpoint /chemin/modele.psckpt \
  --opponent random --max-eligible-pairs 64 \
  --start-horizon 64 --sampling-temperature 1 \
  --sampling-uniform-mix 0.05 \
  --lower-elo -100 --elo0 0 --elo1 100
```

Chaque lot est publié atomiquement avec le journal complet de ses parties. La
reprise continue au prochain identifiant ; une paire interrompue reste
consultable mais ne compte ni comme nul ni dans l'Elo. Le résultat conserve le
score, son équivalent Elo logistique descriptif, l'écart candidat–adversaire
du modèle Bradley–Terry–Davidson avec incertitude par paire lorsqu'il est fini,
et la décision GSPRT entre les hypothèses configurées. `--lower-elo` active
un second test sur les mêmes paires : `lower-elo ↔ elo0` complète alors le test
`elo0 ↔ elo1`. Une force extérieure à la fenêtre est établie par la borne
correspondante ; le centre n'est déclaré soutenu que lorsque les deux tests le
préfèrent à leur extrême. Tous ces chiffres
sont propres au protocole de départ et à la politique inscrits dans l'identité.

```sh
target/release/paisho-evaluate --verify /chemin/evaluation
```

Cette vérification relit les manifestes, rejoue les parties et recalcule le
rapport sans lancer MPSGraph.

## Boucle générationnelle reprenable

`paisho-generations` relie désormais, dans cet ordre, la collecte acteur, le
learner PPO terminal et le duel de promotion. Il faut d'abord construire le
service Swift et les trois exécutables Rust voisins :

```sh
sdk_path=$(xcrun --sdk macosx --show-sdk-path)
SDKROOT="$sdk_path" swift build -c release \
  --package-path apple/paisho-mpsgraph --sdk "$sdk_path"
cargo build --release -p paisho-train --bins

target/release/paisho-generations \
  --campaign-dir /chemin/campagne \
  --target-generation 10 \
  --opponent random \
  --curriculum-dir /chemin/campagne/curriculum
```

La cible de génération est absolue. Chaque génération possède un plan scellé,
puis au plus quatre publications immuables : acteur, learner, promotion et
issue. À l'ouverture, une relance vérifie les empreintes et manifestes des
artefacts référencés, puis réutilise chaque étape déjà publiée ; l'audit
sémantique qui rejoue toutes les parties reste disponible séparément. Une
génération interrompue relit son plan scellé, indépendamment des nouvelles
options, et n'exige que les exécutables des étapes encore absentes. Un
checkpoint sans commit reste traité par le learner selon son protocole
transactionnel. Une promotion indéterminée conserve le champion antérieur mais
poursuit l'apprentissage depuis le candidat ; un rejet explicite revient au
champion. Les rôles `latest`, `candidate`, `champion`, `training` et `best` sont
toujours recalculés depuis cette chaîne, sans pointeur mutable susceptible
d'être perdu.
`best` désigne pour l'instant le champion ; les jalons Elo `milestone` viendront
avec la ligue neuronale.

Les réglages par défaut utilisent tous les workers CPU disponibles, les classes
MPSGraph `64×8, 128×4, 1024×1`, les débuts neutres à horizon 64, 100 pas PPO et
un duel SPRT échantillonné à température 1 avec 5 % de mélange uniforme,
plafonné à 400 paires admissibles sur 800 tentées. Le duel reçoit par défaut
son propre flux de départs neutres à horizon 64 ; sa graine et ses limites sont distinctes de
celles des acteurs et scellées dans le plan. L'adversaire peut être `random`,
`site`, `mcts:8` jusqu'à `mcts:512`, ou `self`. Le choix est scellé par
génération. Sans `--curriculum-dir`, ce choix reste manuel.

Avec `--curriculum-dir`, la valeur initiale de `--opponent` amorce l'échelle
`random → site → mcts:8 → mcts:32 → mcts:128 → mcts:512 → self`. Après chaque
génération achevée, le checkpoint actif passe par `paisho-evaluate` dans une
archive distincte du replay. Par défaut, les mêmes paires alimentent les tests
`−100 ↔ 0` et `0 ↔ +100` Elo. Chaque comparaison reçoit par défaut 2,5 %
d'erreur de chaque type afin que les deux sorties extérieures ne prétendent pas
ensemble au risque nominal d'un seul test. Une borne extérieure soutenue provoque une
avance ou un retour ; un centre soutenu ou un budget indécis maintient le
palier. Au plancher aléatoire, une borne basse maintient naturellement le
palier. En auto-jeu, la promotion candidat/champion reste le sélecteur et MCTS
n'est plus une source de récompense.

Chaque décision est publiée après son évaluation avec le checkpoint, le palier,
le résumé Elo et les empreintes de l'archive. Un arrêt après la génération,
pendant l'évaluation ou juste avant la décision reprend respectivement à
l'évaluation, au prochain lot de paires ou à la publication de la décision.
Les paramètres futurs peuvent changer : leur empreinte ouvre une nouvelle
archive de protocole sans réécrire les preuves antérieures. L'audit complet se
lance sans entraînement avec :

```sh
target/release/paisho-generations \
  --verify-curriculum /chemin/campagne/curriculum
```

Le branchement et sa reprise ont été contrôlés sur la campagne existante ; une
campagne prolongée doit encore mesurer si la fenêtre par défaut fournit le bon
rythme de changement d'adversaire.

## Tableau de bord local

`paisho-dashboard` produit une petite interface HTML en lecture seule à partir
des plans, étapes, rapports acteurs, promotions et décisions Elo déjà scellés.
Elle affiche le nombre de générations, les parties tentées et exploitables par
fonction, les exemples PPO, les checkpoints, le palier courant, le dernier Elo
contextuel et le meilleur point de chaque série strictement comparable.

```sh
target/release/paisho-dashboard \
  --curriculum-dir training-runs/pure-curriculum-001/curriculum \
  --watch-seconds 60
```

Par défaut, le fichier est écrit à côté du répertoire global de campagne, par
exemple `training-runs/pure-curriculum-001-dashboard.html`, jamais dans les
archives immuables. Le mode de surveillance ne lance, n'arrête et ne modifie
aucun entraînement ; il régénère seulement cette vue et celle-ci se recharge au
même rythme. Une vue ponctuelle précédée d'un rejeu complet peut être produite
avec `--verify` à la place de `--watch-seconds`.

Le record affiché en tête appartient toujours au même adversaire et au même
protocole que la dernière mesure. Les records d'autres horizons ou adversaires
restent dans des séries distinctes. Tant qu'aucun match-pont humain n'existe,
l'interface marque explicitement l'Elo The Garden Gate comme non calibré.

## Contrôle local et reprise après redémarrage

Le tableau HTML précédent demeure une projection en lecture seule. Pour le
piloter sans lui donner accès aux archives, `tools/paisho_control.py` l'encapsule
dans un serveur limité à `127.0.0.1`, avec état de processus, progression par
checkpoint, journal et boutons pause/reprise. Le contrôleur lance la commande
dans un groupe de processus dédié : la pause suspend donc aussi ses services
MPSGraph, tandis que la reprise continue en mémoire.

L'état souhaité est fsyncé séparément des résultats. Un LaunchAgent utilisateur
peut restaurer le contrôleur après connexion : une tâche marquée en cours est
relancée avec ses arguments exacts et bénéficie alors des mécanismes de reprise
du learner ou de `paisho-generations`; une tâche marquée en pause ne redémarre
pas avant action de l'utilisateur. Le contrôleur ne fabrique ni checkpoint ni
résultat Elo et n'altère aucun artefact scellé.

## Parties à rejouer dans l'extension WordPress

La console configurée propose `/examples` : téléchargement ou copie du **PSR**
complet, directement utilisable par le lecteur existant. Aucun JSON à importer.
Une partie illustrative est sélectionnée par génération de collecte, avec
préférence pour une victoire du réseau, puis le score terminal. Le préfixe
aléatoire éventuel est conservé, sans l'attribuer au réseau.

Export hors ligne, sans GPU :

```sh
cargo run --release -p paisho-train --bin paisho-export-games -- \
  --snapshot /absolute/path/snapshot.psrsnap --generation 59 \
  --output /absolute/path/examples
```

Résultat : `generation-00000000000000000059/best-game.psr`, accompagné de
métadonnées internes de vérification. Si le snapshot contient plusieurs
producteurs Behavior, préciser `--behavior-producer HASH`.
La boucle `paisho-live` exporte au moment de la collecte ; la console peut
rattraper les campagnes historiques avec un seul worker CPU de fond via
`dashboard.examples_directory`, `examples_campaign_directory` et
`examples_exporter`. Aucun export lourd sur le thread des requêtes HTTP.

# Client Rust pour MPSGraph

Ce crate relie les positions et lots d'entraînement du moteur Rust au service
Swift MPSGraph persistant. Il ne contient ni poids entraînés, ni politique de
jeu, ni stockage de replay.

Le protocole binaire `PSI V1` est cadré par une longueur `UInt64` little-endian.
Une requête transporte un identifiant, une forme fixe, les tenseurs
`StateEncodingV1` et les adresses compactes de chaque coup légal. La réponse
renvoie les probabilités de politique remises dans l'ordre du moteur et les
trois probabilités victoire/nul/défaite. Les deux côtés contrôlent forme,
identifiant, fin de trame, valeurs finies, distributions et padding.

Le protocole `PST V1` ajoute les cibles de politique et WDL, le taux
d'apprentissage, l'étape Adam attendue, le snapshot de replay et sa plage
d'indices. Le service confirme l'étape et l'index suivants et refuse de changer
silencieusement de snapshot au milieu d'une exécution ou après restauration.
Le protocole `PST V2` transporte à la place l'objectif PPO terminal : indice et
probabilité du coup réellement échantillonné, classe WDL finale, valeur de
l'acteur gelé et paramètres de la même transformation
température/exploration. `MpsGraphProcess::train_terminal_ppo` vérifie sa forme,
sa réponse et sa progression comme pour le mode supervisé.
Le protocole `PSC V1` demande ensuite une publication atomique du modèle, de
l'état Adam, du scheduler et des curseurs. Le client recalcule le SHA-256 du
fichier avant d'accepter l'acquittement. Une requête dont l'acquittement a été
perdu peut être rejouée : seul un fichier strictement identique est alors
accepté, sans jamais écraser un contenu différent.

`PSG1` est un petit message JSON dans le même framing :
`MpsGraphProcess::begin_new_generation` change explicitement le snapshot,
remet son index à zéro et conserve poids/Adam/pas global dans le processus.
L'ancienne empreinte et le pas doivent correspondre. Ce pont est testé, mais
la boucle générationnelle ne l'utilise pas encore ; il ne fournit pas à lui
seul la sauvegarde tous les cinq cycles ni la transmission aux acteurs.
Le smoke `training_cycle_smoke` exécute deux cycles PPO sans checkpoint ;
le lancer avec le wrapper de pause/reprise comme les autres essais GPU.

Construire d'abord le service macOS :

```sh
sdk_path=$(xcrun --sdk macosx --show-sdk-path)
SDKROOT="$sdk_path" swift build --package-path apple/paisho-mpsgraph \
  --sdk "$sdk_path" -c release --product paisho-mpsgraph-service
```

Pour les expériences de chevauchement, le banc `capacity_broker_smoke` expose
`--inflight N` (requêtes en vol, avec les tampons Swift correspondants) et
`--prefetch true|false` (préparation d'un prochain lot sur CPU). Le second
mode utilise une seule inférence en vol ; ce n'est pas une combinaison des
deux stratégies. Défauts : une requête, préchargement désactivé. Les essais
actuels n'établissent aucun gain justifiant d'activer ces options en campagne.

Vérifier un lot réel séquentiel :

```sh
cargo run --release -p paisho-mpsgraph-client --example bridge_smoke -- \
  --preset pure --batch 8 --level 1 --warmup 3 --iterations 20
```

Ou alimenter le courtier depuis des workers concurrents :

```sh
cargo run --release -p paisho-mpsgraph-client --example broker_smoke -- \
  --preset pure --batch 8 --level 1 --jobs 400 --wait-us 5000
```

Ou faire avancer de vraies parties à travers plusieurs classes de forme :

```sh
cargo run --release -p paisho-mpsgraph-client --example capacity_broker_smoke -- \
  --preset pure --classes 64:8,128:4,1024:4 \
  --wide-lanes 1 \
  --warmup-decisions 20 --measured-decisions 300
```

Enfin, traverser un vrai shard, sauvegarder, redémarrer et reprendre Adam :

```sh
cargo run --release -p paisho-mpsgraph-client --example replay_training_smoke -- \
  apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-service 4 8 1024 micro
```

Le contrôle minimal du gradient terminal Rust→Swift est également exécutable :

```sh
cargo run --release -p paisho-mpsgraph-client --example terminal_ppo_smoke -- \
  apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-service
```

Le courtier simple remplit une forme MPSGraph fixe et duplique uniquement une ligne
pour compléter un lot partiel ; les sorties de ce padding sont ignorées. La
taille, l'attente maximale et le nombre d'acteurs sont des paramètres de mesure,
pas des constantes d'entraînement. `CapacityInferenceBroker` route une position
vers la plus petite classe admissible et permet une taille de lot différente par
classe. Les services peuvent tous charger le même checkpoint avec
`--checkpoint PATH`. `launch_with_lanes` peut créer plusieurs services et
courtiers indépendants pour une classe ; les requêtes sont réparties en
rotation et la télémétrie conserve les totaux ainsi que chaque file séparée.
Le banc expose cette dimension avec `--wide-lanes`, sans modifier la taille du
lot de la classe la plus large. Sur la M1 Max et la charge actuelle de dix
producteurs synchrones, le balayage de 1 à 5 files conserve 1 par défaut : 2
est à égalité pratique et 3 à 5 augmentent fortement le padding. Les transports
d'entraînement supervisé et PPO terminal sont fonctionnels ; la commande de
sauvegarde distante l'est également.
L'orchestrateur de longue durée qui les emploie se trouve dans
`paisho-train`.

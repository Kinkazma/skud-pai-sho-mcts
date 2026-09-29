# Replay d'entraînement V1

`paisho-replay` conserve des trajectoires complètes et les politiques qui
servent de cibles. Les tenseurs ne sont pas stockés : le moteur rejoue le
`GameRecord`, puis `paisho-model` régénère l'état canonique et les coups légaux.
Une évolution compatible du moteur ne peut donc pas réaffecter silencieusement
un poids au mauvais coup.

Une politique est clairsemée et indexée par `ActionEncodingV1`. Les coups
absents ont un poids nul ; les entrées présentes doivent être positives,
uniques et totaliser 1. Le producteur et la nature de la cible distinguent
politique de comportement, visites MCTS, enseignant et coup joué one-hot.
Une cible « coup joué » doit nommer exactement le coup du journal. Seules les
décisions explicitement retenues deviennent des exemples.

La valeur terminale suit l'ordre nommé `[win, draw, loss]` du point de vue du
joueur réellement actif avant la décision. Le replay ne suppose jamais que les
joueurs alternent : un bonus d'Harmonie peut donner deux décisions successives
au même camp. Une partie incomplète est refusée au lieu d'être transformée en
nul.

Les shards binaires `PSRPLY01` sont canoniques, couverts par SHA-256, publiés
atomiquement et jamais écrasés. La version 2 peut joindre à une décision de
comportement la valeur WDL déjà prédite par son acteur ; les shards V1 restent
lus et vérifiés dans leur représentation d'origine. Un snapshot texte ordonne
leurs noms, indices, empreintes et nombres de parties/exemples, puis possède sa
propre empreinte.
`ReplaySnapshotV1::verify_directory` relit chaque shard, rejoue les parties et
compare tout le contenu déclaré, tout en refusant qu'un identifiant de partie
apparaisse dans deux shards. Cette empreinte de snapshot est enregistrée dans
le checkpoint MPSGraph V2 avec le curseur de replay.

Après cette validation sémantique de publication,
`verify_directory_integrity` permet aux reprises ordinaires de recalculer en
flux les empreintes exactes sans rejouer chaque décision. Une disparition ou
une altération reste donc détectée, tandis que l'audit sémantique complet reste
disponible à la demande.

`ReplayDatasetV1` charge d'abord les journaux compacts d'un snapshot vérifié.
Il peut ensuite reconstruire tous les exemples en parallèle, groupés par
partie, et les conserver une fois en RAM. `ReplaySamplerV1` applique une
permutation déterministe différente à chaque époque puis ne fait plus que
référencer ce cache. Son état minimal est le hash du snapshot, la graine et le
prochain index absolu. Une reprise de cet état rend exactement la même suite, y
compris lorsqu'un lot traverse une frontière d'époque ; un autre snapshot est
refusé. Préparer un lot ne déplace pas le curseur : le learner doit le confirmer
explicitement après l'acquittement du pas distant.

Pour le PPO terminal, `ReplayDatasetV1::from_snapshot_for_behavior` ne retient
que les décisions `Behavior` d'un producteur exact. Chaque exemple expose
l'indice et la probabilité du coup joué ainsi que le retour terminal depuis la
perspective active. Pour les nouveaux shards V2, il expose aussi la valeur
`P(win)-P(loss)` de l'acteur au moment du coup. Le learner évite ainsi une
seconde inférence sur tout le corpus ; si le corpus V1 est ancien ou incomplet,
la valeur absente est explicitement signalée et peut être recalculée par la
voie compatible. Les décisions gagnantes et perdantes deviennent donc deux
signaux distincts utilisés une fois chacun ; les cibles MCTS, enseignant et
one-hot restent hors de ce dataset.

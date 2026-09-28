# Confronto evaluation LIBERO — aggiornato al 28 settembre 2026

Tutte le quattro evaluation attuali sono complete: **16 suite, 8.000 episodi**.
Ogni suite contiene 10 task × 50 prove = 500 episodi. Le celle riportano
`success rate (successi/500)`; la media di ciascun run è anche il rapporto
fra tutti i successi e i suoi 2.000 episodi.

## Risultati

| Checkpoint valutato | Spatial | Object | Goal | Long (`libero_10`) | Media | Successi totali | Δ vs paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper, tabella 1 aggiornata | 97,0% | 100,0% | 98,2% | 95,2% | **97,60%** | non forniti come log | — |
| Autori unified, EMA dichiarata a 34k | 97,8% (489) | 99,2% (496) | 98,0% (490) | 94,2% (471) | **97,30%** | **1946/2000** | −0,30 pp |
| Autori non-unified, quattro export storici | 97,6% (488) | 99,2% (496) | 95,4% (477) | 92,4% (462) | **96,15%** | **1923/2000** | −1,45 pp |
| Nostro `libero-baseline`, EMA 34k | 97,8% (489) | 98,8% (494) | 96,2% (481) | 91,2% (456) | **96,00%** | **1920/2000** | −1,60 pp |
| Nostro `libero-baseline`, EMA 80k | 98,4% (492) | 98,8% (494) | 95,0% (475) | 91,8% (459) | **96,00%** | **1920/2000** | −1,60 pp |

I valori paper sono quelli pubblicati nel repository ufficiale con il
[commit `6727c87`](https://github.com/H-EmbodVis/TurboVLA/commit/6727c875666f8d5dda8d8cca0043da200738fe73).
La media aritmetica delle quattro suite è esattamente **97,60%** e costituisce
il riferimento della colonna Δ. La precedente tabella arXiv v2 riportava 97,7%.
`pp` significa punti percentuali, non variazione percentuale relativa.

Il checkpoint unified è superiore al valore aggiornato del paper di 0,8 pp su
Spatial, ma è inferiore di 0,8 pp su Object, 0,2 pp su Goal e 1,0 pp su Long.
Non abbiamo riprodotto esattamente tutte le suite.

Fra i nostri 34k e 80k, Spatial guadagna 3 successi e Long 3, Goal ne perde 6,
Object è invariata: **nessun miglioramento medio a 80k**. Entrambi sono
1,30 pp sotto l'unified degli autori. Non abbiamo valutato tutti gli altri
checkpoint: questo confronto non identifica il migliore dell'intero training.
Una sola valutazione per checkpoint/seed non misura la variabilità fra run
e non dimostra equivalenza statistica con il paper.

## Protocollo effettivamente registrato

- Una GPU A40 per job; quattro invocazioni sequenziali, una per suite, nello
  stesso ordine: Spatial, Object, Goal, Long. Nessun array o batching di ambienti.
- 50 prove per task, tutti i task 0–9, seed 7, attesa iniziale 10 step.
- Chunk 12, open-loop 12, immagini 256 px, controllo relativo, BF16, EGL,
  `dinov3_output_hidden_states=true`, video disabilitati, download HF disabilitati.
- Statistiche `libero_all4_stats.json`, chiave `libero_all4_no_noops`;
  `normalize_binary_gripper=auto`, risolto a `False` nei log.
- Backbone locale DINOv3 ViT-B/16 e BERT; Transformers 4.56.0, MuJoCo 2.3.7.
  Configurazione, pacchetti, comando e copia dei sorgenti sono registrati per suite.
- I checkpoint nostri caricano `ema_model_state_dict`. L'unified carica
  l'export ufficiale verificato tramite SHA256, dichiarato EMA dagli autori.
  Per i quattro export storici carichiamo i pesi pubblicati, senza convertirli
  e **senza dedurre che siano EMA** dal solo nome o dai metadati.

La [guida upstream](https://github.com/H-EmbodVis/TurboVLA/blob/b29ab1420baa5c663ec935df513f2012430beb67/experiments/libero/README.md#evaluation)
usa una suite per invocazione; lo scheduling sequenziale è il nostro adattamento
al cluster. Gli autori raccomandano l'unified su tutte le suite e spiegano che
gli export precedenti derivavano dal training misto, non da quattro training
indipendenti: [chiarimento sulla release](https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902).

Il nostro training `libero-baseline` è arrivato a 80k step su 4 GPU, batch
globale **128 = 4 × 8 × 4**, con pretrained pubblici, non con tutti i pesi
inizializzati casualmente. Questo segue i default pubblici e il
[chiarimento degli autori su batch, EMA e selezione checkpoint](https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736).
Non è una replica bit-per-bit del run storico: il paper scrive batch 256 e
indica RTX 4090, mentre qui sono state usate A40.

Il dataset RLDS `*_no_noops` serve al **training**. L'evaluation non legge
le dimostrazioni: usa il simulatore LIBERO, i task/stati iniziali e le statistiche
di normalizzazione. Non serve rigenerare RLDS per eseguire queste evaluation.

## Verifiche e warning

Controllati i 16 `results.json`: task 0–9, 50 episodi ciascuno, somme e percentuali
coerenti. Confrontati anche tutti gli 8.000 record `task/episode/success` nei log
Slurm: ogni coppia task/episodio è presente una volta per suite e i successi
corrispondono ai JSON, anche task per task. Nessuna suite è parziale.

Non risultano traceback o terminazioni fatali nei log esaminati. **Tutti e quattro
i run hanno però un warning MuJoCo nella suite Long**:
`Nan, Inf or huge value in QACC at DOF 9`, tempo simulato `0.5480`.
Il warning non ha interrotto il processo. Da questi log non è possibile
attribuirgli con certezza un episodio o quantificare l'effetto sul successo;
non è dimostrato che spieghi lo scarto dal paper. Non sono stati esclusi o
corretti a posteriori episodi/risultati.

I rispettivi `libero_10/source/MUJOCO_LOG.TXT` sono conservati come evidenza
del run. È stato eliminato soltanto il vecchio omonimo nella root della repository.

## Provenienza e file consultabili

Root risultati: `/mnt/beegfs/gpepe/TurboVLA/results/libero/`.
In ogni directory sotto elencata, ciascuna delle quattro suite contiene
`results.json`, `config.json`, `checkpoint.json`, `execution.json`, `command.txt`,
`git-head.txt`, `pip-freeze.txt` e `source/`. Il solo commit Git non descrive le
modifiche locali: per riprodurre l'esecuzione fa fede anche la copia `source/`.

| Run / directory | Job ID (solo tracciabilità, non nome delle cartelle) | Log Slurm |
|---|---|---|
| [libero-author-chekpt-unified](/mnt/beegfs/gpepe/TurboVLA/results/libero/libero-author-chekpt-unified) | 1929454 | [out](/home/gpepe/ws/logs/turbovla/evaluation/libero-author-chekpt-unified.out), [err](/home/gpepe/ws/logs/turbovla/evaluation/libero-author-chekpt-unified.err) |
| [libero-author-chekpt-non-unified](/mnt/beegfs/gpepe/TurboVLA/results/libero/libero-author-chekpt-non-unified) | 1929455 | [out](/home/gpepe/ws/logs/turbovla/evaluation/libero-author-chekpt-non-unified.out), [err](/home/gpepe/ws/logs/turbovla/evaluation/libero-author-chekpt-non-unified.err) |
| [libero-baseline-34k](/mnt/beegfs/gpepe/TurboVLA/results/libero/libero-baseline-34k) | 1927109 | [out](/home/gpepe/ws/logs/turbovla/evaluation/libero-baseline-34k.out), [err](/home/gpepe/ws/logs/turbovla/evaluation/libero-baseline-34k.err) |
| [libero-baseline-80k](/mnt/beegfs/gpepe/TurboVLA/results/libero/libero-baseline-80k) | 1927110 | [out](/home/gpepe/ws/logs/turbovla/evaluation/libero-baseline-80k.out), [err](/home/gpepe/ws/logs/turbovla/evaluation/libero-baseline-80k.err) |

Le percentuali provengono dai JSON originali, non dal parsing delle barre tqdm.
I file `.err` contengono anche progressi e messaggi INFO: non sono solo errori.

### Identità dei pesi

Percorsi relativi a `$SCRATCH_FLASH/TurboVLA`:

| Pesi | Percorso | SHA256 |
|---|---|---|
| Unified, tutte le suite | `pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth` | `d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab` |
| Legacy Spatial, 51k | `pretrained/TurboVLA/checkpoints/libero/spatial.pth` | `a7c3faa825a6c68d365df0647c39845c3a7bb553e1e24be3729b76de22f703fa` |
| Legacy Object, 75k | `pretrained/TurboVLA/checkpoints/libero/object.pth` | `787c01bd8b328a5948b756aab92f8058a1e0802845a0e1f24506291b9cda59cf` |
| Legacy Goal, 49k | `pretrained/TurboVLA/checkpoints/libero/goal.pth` | `60070c9b1735e34ea91143f48e2dea2ed8e3b179d586801529ab253fdaa21a8a` |
| Legacy Long, 60k | `pretrained/TurboVLA/checkpoints/libero/long.pth` | `96123b539df63573860b853d37964f41ae47f1b9cdddd2431fb9095eea4945f3` |
| Baseline 34k | `training/libero/libero-baseline/checkpoints/turbovla_step_34000.pth` | `8473fce9f36a97af08ede8bf008f33f5205b536307b6175817ede70c5ccb638e` |
| Baseline 80k | `training/libero/libero-baseline/checkpoints/turbovla_step_80000.pth` | `c0fa3d5198c98fa6a520a9cab0f0a7a0530b0fe1795c470e49995f432b541d1a` |

Revisioni Hugging Face fissate di `H-EmbodVis/TurboVLA`: unified
`cb5300544693013164c4bb251a13036002a55c81`; legacy
`f7b0f53afa248408d20748f2579e446c7ce4119e`.
Gli hash dei checkpoint baseline nella tabella sono quelli registrati nelle
evaluation; gli hash dei pesi ufficiali sono stati anche ricalcolati nell'audit odierno.

Le evaluation storiche eliminate su richiesta non sono mescolate con questi run.
La vecchia directory documentale `scripts/cluster/archive` è stata rimossa perché
non faceva più parte del workflow operativo.

# Training LIBERO: ricetta rilasciata e audit delle modifiche

Audit del 6 settembre 2026. Ambito: **LIBERO**, non RoboTwin.
Upstream verificato via API GitHub: `b29ab1420baa5c663ec935df513f2012430beb67`.
L'ultimo aggiornamento del training è `ced2b0c465c201ac0c6d41a6855b0c0040360df6`
(2 settembre: batch size ed evaluation EMA). Non sono emersi commit successivi
da integrare. Non sono stati lanciati nuovi training da questa guida.

## 1. Cosa possiamo chiamare riproduzione

Il [paper v2, §5.2](https://arxiv.org/html/2607.27205v2#S5.SS2)
scrive batch globale **256**. La
[guida LIBERO upstream](https://github.com/H-EmbodVis/TurboVLA/blob/b29ab1420baa5c663ec935df513f2012430beb67/experiments/libero/README.md)
ora prescrive **128**. Nelle risposte degli autori alle
[issue #12](https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736)
e [#13](https://github.com/H-EmbodVis/TurboVLA/issues/13#issuecomment-5504922032)
la raccomandazione è esplicita: batch 128, pesi EMA, checkpoint selezionati
mediante success rate LIBERO. Il nuovo checkpoint pubblico è dichiarato EMA a
34k step. L'[issue #11](https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902)
conferma un unico training congiunto, non quattro training separati.

Di conseguenza il launcher offre:

| Profilo | GPU | Batch/GPU | Accumuli | Batch globale | Interpretazione |
|---|---:|---:|---:|---:|---|
| `upstream128` (default) | 4 | 8 | 4 | 128 | Ricetta attualmente raccomandata dagli autori |
| `paper256` | 4 | 16 | 4 | 256 | Batch del paper/pre-correzione, resto del codice attuale |

`paper256` **non** ricostruisce automaticamente ogni dettaglio del run storico.
Non possiamo affermare contemporaneamente identità al testo del paper e alla
ricetta aggiornata. Per confrontarsi con il nuovo checkpoint partire da
`upstream128`. Il paper usa quattro RTX 4090; qui sono quattro A40.
Non è una riproduzione bit per bit né una garanzia del 97.7%.

## 2. Modifiche rispetto al codice originale

Verificabili con `git diff upstream/main -- turbovla scripts/robotwin/run_policy_server.sh`.

### Caricamento del checkpoint ufficiale: `turbovla/evaluation/policy.py`

Upstream richiede `ema_model_state_dict`, ma il file pubblico di settembre
contiene `model_state_dict` e `model_config`. Abbiamo aggiunto un'eccezione
**solo per il file con SHA256**
`d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab`.
Se manca EMA, si può usare quella chiave soltanto per questo export verificato.
Non si convertono tensori; resta `load_state_dict(strict=True)`.
Se EMA esiste, ha sempre precedenza. Il fatto che l'export rappresenti EMA è
una dichiarazione degli autori; l'hash certifica l'identità del file, non la sua
storia di training. I checkpoint raw arbitrari rimangono rifiutati.
Questa patch non influenza il training.

### Ripresa EMA: `turbovla/training/pi05.py` e `trainer.py`

**Correzione della nostra precedente analisi:** EMA era già implementata nel
percorso ufficiale. Non l'abbiamo introdotta noi. Il trainer pubblico importa
`train_mixed`, che importa `pi05`, che sostituisce optimizer e salvataggio del
trainer di base. Fermarsi alla lettura del solo `trainer.py` era insufficiente.

Il problema reale era la ripresa: gli accumulatori `_ema_params` non fanno
parte dello `state_dict()` standard di AdamW. Senza il nostro fix, alla ripresa
venivano ricreati dai pesi raw anziché caricati da EMA.

La nuova `restore_ema()` verifica nomi, forme e decay, ripristina i tensori
salvati e viene chiamata con `resume_mode=all`, dopo optimizer e scheduler.
Il salvataggio registra il decay effettivo dell'optimizer.
Non cambia l'aggiornamento di un training fresco con decay 0.999.
Non è ancora un fix upstream. Stato RNG e posizione dell'iteratore dati non
sono salvati/ripristinati: una ripresa non equivale bit per bit al run continuo.

### Adattamenti al cluster

`scripts/cluster/` aggiunge ambienti separati, download a revisioni fissate,
percorsi scratch, esecuzione offline, job Slurm, controlli e report. Non
sostituisce architettura, loss, dataset loader o normalizzazione upstream.
I test di regressione sono in `tests/`.
Esiste inoltre una modifica locale preesistente a
`scripts/robotwin/run_policy_server.sh`: usa BERT/DINO locali e disabilita il
preload di inizializzazione quando carica il checkpoint completo per evaluation.
Non è stata modificata in questo audit e non è nel percorso LIBERO.

## 3. EMA e il nome pi05

EMA significa **Exponential Moving Average**, media mobile esponenziale dei
pesi. Dopo ogni aggiornamento dell'optimizer:

```text
pesi_EMA = 0.999 * pesi_EMA_precedenti + 0.001 * pesi_appena_aggiornati
```

Il gradiente aggiorna i pesi normali; in parallelo si mantiene questa copia
smussata dei parametri allenabili. Può ridurre le oscillazioni tra checkpoint;
non garantisce che ogni valutazione migliori. I parametri congelati sono
inclusi nel checkpoint completo senza necessità di una media separata.
EMA dei pesi non è la media della loss, né dei gradienti, né un ensemble di
modelli da eseguire insieme durante l'inferenza.

`pi05.py` è un nome **già presente upstream**: la sua docstring descrive
"pi0.5 optimizer knobs". Implementa AdamW con beta `(0.9, 0.95)`, epsilon
`1e-8`, EMA e configurazione della precisione DINO. Non importa il modello
π0.5 e non carica suoi pesi. La rete allenata resta TurboVLA, con loss L1.
Non deduciamo dal nome che l'intera ricetta sia identica a quella di π0.5.

## 4. Pesi iniziali, dati e preprocessing

Root: `${SCRATCH_FLASH}/TurboVLA`, qui `/mnt/beegfs/gpepe/TurboVLA`.

| Componente | Asset locale | Origine e revisione fissata |
|---|---|---|
| Visione | `models/dinov3-vitb16` | `facebook/dinov3-vitb16-pretrain-lvd1689m` @ `5931719e67bbdb9737e363e781fb0c67687896bc` |
| Linguaggio | `models/bert-base-uncased` | `google-bert/bert-base-uncased` @ `86b5e0934494bd15c9632b12f734a8a67f723594` |
| Interazione/proiezione | `models/groundingdino/groundingdino_swint_ogc.pth` | `ShilongLiu/GroundingDINO` @ `84311ae61139581d0e62eca0bad610ad14e70aef` |
| Dimostrazioni | `datasets/libero` | `openvla/modified_libero_rlds` @ `a7c9ae18499b6eea8a32f78a9302327b752b1b5f` |

Il [README GroundingDINO](https://github.com/IDEA-Research/GroundingDINO#-checkpoints)
collega sia il download GitHub sia questo mirror HF per Swin-T OGC.
SHA256 del peso: `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`.
I controlli fissano anche gli hash dei pesi DINO/BERT e dei due JSON rilasciati.
Sono i tipi di pesi iniziali nominati nella ricetta, alle nostre revisioni
pubbliche fissate. Gli autori non pubblicano un manifest completo del loro
run storico: non possiamo provare che ogni byte delle loro copie fosse uguale.

L'inizializzazione ufficiale usa:

- DINOv3 pretrained come backbone visivo, completamente allenabile;
- BERT pretrained congelato, eseguito online;
- `transformer.encoder.fusion_layers.*` e `text_layers.*` di GroundingDINO:
  nel test reale, **180 tensori** caricati nel modulo di interazione;
- `feat_map.*` di GroundingDINO: **2 tensori** per la proiezione testuale;
- gli altri componenti, incluso il decoder delle azioni, inizializzati dal
  costruttore del modello con seed 42, non da un checkpoint TurboVLA allenato.

Non usiamo il backbone Swin o il detector completo di GroundingDINO. I log
del test mostrano zero mapping saltati per nomi o forme. Il training LIBERO
"da zero" significa qui **da questi pretrained iniziali**, non da pesi
interamente casuali, né fine-tuning del checkpoint TurboVLA a 34k.

I dati sono già la versione `no_noops` RLDS richiamata dal paper. Non passiamo
per una conversione LeRobot e non rilanciamo la rigenerazione: lo script
`regenerate_libero_no_noops.py` serve per riprodurre la trasformazione a partire
dai dati originali, rimuovendo azioni no-op e replay non riusciti, prima della
conversione TFDS/RLDS. Non è un passaggio aggiuntivo sui dati già modificati.

| Suite | Episodi da metadata | Shard verificati |
|---|---:|---:|
| Long (`libero_10_no_noops`) | 379 | 32 |
| Goal | 428 | 16 |
| Object | 454 | 32 |
| Spatial | 432 | 16 |
| Totale | 1693 | 96 |

Il preflight ha riletto e verificato tutti gli shard contro gli hash dei
metadata di download HF alla revisione fissata, oltre a features e dataset_info.
Questo verifica l'integrità dei file scaricati; non equivale a riesaminare
semanticamente ogni episodio o a confrontarli con file privati degli autori.

Manteniamo `libero_all4_stats.json`, chiave `libero_all4_no_noops`, senza
ricalcolo. Il loader usa mean/std della propriocezione e min/max delle azioni;
`normalize_binary_gripper=auto` risolve a false con queste statistiche.
`online_text_layout.json` è metadata di padding/tokenizzazione, non embedding
testuali precalcolati. Il mixed loader concatena le suite e applica lo shuffle
ufficiale; non abbiamo introdotto un campionamento uniforme per task.
L'evaluation usa il simulatore, task e stati iniziali LIBERO: non legge queste
dimostrazioni RLDS per eseguire gli episodi.

## 5. Parametri del launcher

Tutti i flag sono espliciti in `libero_training_recipe.sh` e confrontati nei
test con il parser pubblico. I dettagli non menzionati nel paper provengono
dai default del codice rilasciato, non da nostre ipotesi sul run privato.

| Parametro | Valore |
|---|---|
| Training | Un solo modello, quattro suite insieme, 80.000 optimizer step |
| Warmup / LR | 10.000 step / `5e-5` per DINO e resto allenabile |
| Optimizer / EMA | AdamW, beta 0.9/0.95, epsilon 1e-8, EMA 0.999 |
| Weight decay / clipping | 1e-10 sui gruppi decay, 0 sui no-decay / norma 1.0 |
| Precisione | Parametri policy FP32; autocast BF16 nel forward DINO |
| Seed | 42 per training; l'evaluation usa 7 |
| Modello | Due viste 256×256, hidden 256, sei layer di interazione |
| Azioni / stato | Chunk 12, azioni 7D, stato 8D, due state token |
| Loader | Quattro worker **per processo GPU**, buffer episodi 512, step buffer 64 |
| Salvataggio | Ogni 1.000 step e alla fine; pesi raw, EMA, optimizer, scheduler |

Attenzione: `min_lr_ratio=1.0` è il default upstream. Anche se la funzione si
chiama scheduler cosine, con rapporto 1 il LR resta costante dopo il warmup.
Non abbiamo aggiunto un decadimento cosine non presente nella ricetta attiva.

Ambiente controllato: torch 2.3.1+cu121, torchvision 0.18.1+cu121,
transformers 4.56.0, TF 2.20.0, TFDS 4.9.3, NumPy 1.26.4.
La versione Transformers segue anche il chiarimento degli autori nella
[issue #4](https://github.com/H-EmbodVis/TurboVLA/issues/4#issuecomment-5217436771).
L'issue #14 chiede perché DINO non sia congelato, ma non ha risposte al momento
dell'audit: non costituisce una correzione alla guida full-unfreeze.

## 6. Lancio, verifica e ripresa

File nuovo: `scripts/cluster/train_libero.sbatch`.
Un nodo, quattro GPU A40, 32 CPU, 256 GB RAM, limite 20:50:00. Queste sono
richieste di risorse del cluster, non iperparametri del paper. Nessun array.
Le quattro GPU collaborano via DDP nello **stesso** training: non sono quattro
training indipendenti come i quattro job delle suite in evaluation.

Prima eseguire il breve controllo DDP, non ancora effettuato:

```bash
cd /home/gpepe/ws/TurboVLA
mkdir -p /home/gpepe/ws/logs
sbatch --time=02:00:00 scripts/cluster/train_libero.sbatch upstream128 ddp-smoke
```

Questo fa tre optimizer step usando gli stessi quattro processi, batch,
worker e schedule del training lungo. Non misura il successo LIBERO.
Non sostituisce il test precedente a una GPU, già superato dal job 1921745.

Dopo aver verificato il job breve, training completo consigliato:

```bash
sbatch scripts/cluster/train_libero.sbatch upstream128 full
```

Solo se si vuole studiare separatamente il batch scritto nel paper:

```bash
sbatch scripts/cluster/train_libero.sbatch paper256 full
```

Non lanciare entrambi automaticamente. Il profilo a batch 16/GPU richiede una
verifica di memoria propria. Il training lungo non è stato cronometrato e non
è garantito che 80k step entrino in 20:50 ore.

Output predefiniti:

```text
/home/gpepe/ws/logs/turbovla-libero-train_JOBID.out/.err
$SCRATCH_FLASH/TurboVLA/training/libero_upstream128_full/job-JOBID/
  checkpoints/turbovla_step_1000.pth ... turbovla_step_80000.pth
  assets.json
  recipe.txt
  attempts/JOBID/source/          copia del codice effettivamente eseguito
  attempts/JOBID/command.sh       comando espanso, registrato per audit
  attempts/JOBID/train.log
  attempts/JOBID/pip-freeze.txt
  attempts/JOBID/gpus.txt
```

Il launcher non scarica, non sovrascrive run esistenti e non invia job a catena.
Un lock impedisce due training contemporanei nella stessa directory. Prima di
partire controlla hash degli asset, versioni principali e quattro GPU visibili.
Ogni tentativo usa una copia del codice, evitando import dal worktree modificabile.
I checkpoint del test occupavano circa 2 GB ciascuno: conservare 80 checkpoint
può richiedere circa 160 GB, oltre a dati e altri risultati. Non vengono cancellati.

In caso di time limit, controllare prima che l'ultimo checkpoint sia completo.
Lo upstream scrive direttamente il file finale, non con rename atomico: un job
ucciso durante il salvataggio può lasciare un file incompleto. La ripresa usa
il checkpoint numericamente più recente; non salta automaticamente file corrotti.
Non cancellare o rinominare checkpoint senza un controllo esplicito.

Ripresa manuale del **medesimo run full**, dopo verifica:

```bash
sbatch scripts/cluster/train_libero.sbatch upstream128 full resume \
  /mnt/beegfs/gpepe/TurboVLA/training/libero_upstream128_full/job-JOBID_ORIGINALE
```

Ripristina raw model, optimizer, scheduler ed EMA. Rifiuta cambiamenti a
ricetta, codice eseguito, asset o versioni principali. Non promuovere il run
`ddp-smoke` a `full`: il training di riproduzione parte nuovo, con seed 42.

## 7. Come confrontare i risultati

Conservare almeno i checkpoint a 34k e 80k, ma non assumere che 80k sia il
migliore. Gli autori confrontano success rate di checkpoint intermedi, non
solo loss. Nel nostro run 34k può non essere il massimo; la procedura completa
di selezione del run storico (tutti i candidati/risultati) non è rilasciata.
Per ogni confronto usare **lo stesso checkpoint sulle quattro suite**, 50
episodi per task, e riportare lo step senza selezionare un best diverso per suite.

Per esempio, quando il checkpoint a 34k è stato salvato e verificato:

```bash
export TURBOVLA_LIBERO_CHECKPOINT=/mnt/beegfs/gpepe/TurboVLA/training/libero_upstream128_full/job-JOBID_ORIGINALE/checkpoints/turbovla_step_34000.pth
export TURBOVLA_LIBERO_EVAL_RUN_ID=train_JOBID_ORIGINALE_step34000
export TURBOVLA_LIBERO_EVAL_ROOT=/mnt/beegfs/gpepe/TurboVLA/results/libero_training_eval/train_JOBID_ORIGINALE_step34000
bash scripts/cluster/submit_libero_official.sh
```

Questo riutilizza il protocollo delle nostre evaluation e invia quattro job
indipendenti da una GPU. L'override del checkpoint è essenziale: senza di esso
il vecchio helper punta alla release storica. I nuovi checkpoint di training
contengono EMA e non richiedono l'eccezione hash dell'export ufficiale.
Nessuna evaluation viene inviata automaticamente dal launcher di training.

## 8. Verifiche effettuate e limiti

- Test parser: il profilo `upstream128` equivale ai default ufficiali; la
  variante `paper256` cambia soltanto batch; lo smoke conserva schedule e worker.
- Audit offline completo: tutti i 96 shard attesi e gli asset modello verificati.
- Test precedente reale a una GPU: due step + ripresa + terzo step, EMA
  ripristinata, checkpoint caricato rigorosamente e quattro rollout completati.
- Ancora da verificare: DDP a quattro GPU con quattro worker per rank,
  throughput, consumo di memoria e convergenza del training lungo.
- Questa preparazione non garantisce di riprodurre il numero esatto del paper.
  La baseline pubblica misurata sul cluster rimane 97.30%, contro 97.65%
  (arrotondato 97.7%) della tabella del paper.

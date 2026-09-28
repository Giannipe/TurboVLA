# TurboVLA sul cluster — quattro file operativi

La repository resta in `/home/gpepe/ws/TurboVLA`; dati, pesi, checkpoint e risultati
vanno in `$SCRATCH_FLASH/TurboVLA`, qui `/mnt/beegfs/gpepe/TurboVLA`.
I nuovi log vanno in **`/home/gpepe/ws/logs/turbovla/`**: `/ws` da solo non
esiste su questo cluster. Tutte le operazioni si lanciano con **`sbatch`**.

Per la nuova estensione **LIBERO+ solo evaluation**, con ambiente separato
`turbovla-liberoplus`, vedere [LIBEROPLUS.md](LIBEROPLUS.md). I launcher
`assets.sh`, `envs.sh` e `evaluate.sh` accettano `--benchmark liberoplus`;
`train.sh` no. In `assets.sh` e `envs.sh`, `--benchmark all` resta LIBERO +
RoboTwin, non include LIBERO+. In evaluation usare un benchmark esplicito e
`--suite all` per le quattro suite LIBERO/LIBERO+.
Stato LIBERO+: asset scaricati, estratti e verificati; ambiente isolato installato
e controlli CPU passati (job 1930349, exit 0), 57 test locali passati.
Evaluation GPU completata sulle quattro suite: **5.469/10.030 successi (54,53%)**.
Risultati per suite, tentativi parziali esclusi e limite noto grayscale sono nella
guida.

| File | Funzione |
|---|---|
| [envs.sh](envs.sh) | Crea/aggiorna gli ambienti LIBERO/RoboTwin; per LIBERO+ crea un clone separato e installa il simulatore già scaricato |
| [assets.sh](assets.sh) | Scarica, riprende e verifica modelli, checkpoint, dataset e asset dei simulatori |
| [evaluate.sh](evaluate.sh) | Valuta un checkpoint esplicito, con i default del protocollo rilasciato |
| [train.sh](train.sh) | Avvia il training congiunto con default documentati e configurazione modificabile |

Ogni `.sh` contiene le proprie direttive `#SBATCH`: non servono submitter o
altri file `.sbatch`. Il suffisso `.sh` è valido per Slurm. I quattro Python in
`_internal/` sono implementazione condivisa, **non ulteriori comandi operativi**.
Non esiste più `--submit`: è `sbatch` a inviare il job. Con `bash` sono consentiti
soltanto `--help` e `--dry-run`, per evitare esecuzioni accidentali sul login node.
`--dry-run` mostra il piano senza scrivere, scaricare o inviare job.

Prima della prima sottomissione:

```bash
cd /home/gpepe/ws/TurboVLA
mkdir -p /home/gpepe/ws/logs/turbovla/{training,evaluation,assets,envs}
```

La cartella dei log deve esistere **prima di `sbatch`**, perché Slurm apre out/err
prima di eseguire lo script. I launcher selezionano l'interprete Conda del
benchmark senza richiedere `conda activate` sul login node. Per un altro checkout
esportare `TURBOVLA_REPO`; per un interprete non standard `TURBOVLA_PYTHON`.
`envs.sh` usa `TURBOVLA_CONDA_EXE`, default `/home/gpepe/miniconda3/bin/conda`.

Scegli un nome leggibile con **`sbatch --job-name=...`**. Lo stesso nome viene
usato nei log e, salvo `--output`, nella directory del nuovo training/evaluation.
Non aggiungiamo job ID, hash o timestamp ai nomi. Usare lettere, cifre, punti,
trattini e underscore, senza spazi o slash. Per esempio `libero-baseline` o
`libero-object-checkpoint-paper`. Un esperimento diverso deve avere un nome nuovo.
In anteprima senza Slurm il fallback è `turbovla-train-libero` o
`turbovla-evaluate-libero`; per simulare il nome usare
`SLURM_JOB_NAME=libero-baseline bash scripts/cluster/train.sh --benchmark libero --dry-run`.

La vecchia directory `archive` è stata rimossa: non conteneva componenti usati
dal workflow operativo. Le pulizie sono documentate nelle sezioni 6, 8 e 9.
Dataset e pesi ufficiali
sono conservati; alcuni risultati e run storici sono stati eliminati su richiesta.
Le quattro evaluation attuali sono complete e confrontate in
[LIBERO_RESULTS.md](LIBERO_RESULTS.md). I test locali sono spiegati in
[tests/README.md](../../tests/README.md).

## 1. Quale ricetta stiamo seguendo

Riferimento verificato: upstream `6727c875666f8d5dda8d8cca0043da200738fe73`.

- [README principale](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/README.md)
  e [guida LIBERO](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/experiments/libero/README.md).
- [Guida RoboTwin](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/experiments/robotwin/README.md)
  e [config task-balanced](https://github.com/H-EmbodVis/TurboVLA/blob/6727c875666f8d5dda8d8cca0043da200738fe73/experiments/robotwin/configs/taskbalanced_all50.yaml).
- [Paper v2, §5](https://arxiv.org/html/2607.27205v2#S5) e chiarimenti degli autori:
  [#11](https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902),
  [#12](https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736),
  [#4](https://github.com/H-EmbodVis/TurboVLA/issues/4#issuecomment-5217436771).

Le discrepanze non vengono nascoste:

| Punto | Default scelto | Motivazione |
|---|---|---|
| Batch LIBERO | **128 = 4 GPU × 8 × 4 accumuli** | README aggiornato e risposta degli autori; il paper v2 scrive 256 |
| Training LIBERO | **40k** optimizer step, warmup 10k | Documentazione ufficiale aggiornata in `6727c87`; il parser conserva ancora il default storico 80k e il checkpoint pubblico è dichiarato EMA a **34k** |
| Training RoboTwin | **55k** step, warmup 1k | Paper e comando del README principale; script/YAML generici hanno 100k |
| Evaluation RoboTwin | **100** prove/task | Risultato clean50 pubblicato; il wrapper upstream da solo imposta 20 |
| Transformers LIBERO | **4.56.0** | Gli autori segnalano differenze nelle feature DINO con versioni successive |

Questi sono default per riprodurre la **ricetta pubblica raccomandata**, non
una promessa del risultato numerico esatto. Il paper usa RTX 4090, qui A40.
Gli autori non forniscono un manifest completo dei file e del run storico.

## 2. Ambienti Conda

```bash
cd /home/gpepe/ws/TurboVLA
sbatch scripts/cluster/envs.sh --benchmark libero
# Oppure: --benchmark robotwin / --benchmark all
```

`envs.sh` segue `conda create ... python=3.10`, installazione di PyTorch
compatibile e `pip install -e ".[libero]"` oppure `.[robotwin]`.
Riusa gli ambienti esistenti senza cancellarli, ma può aggiornare pacchetti:
**non rilanciarlo durante job attivi**. Non è necessario reinstallare ora gli
ambienti già funzionanti soltanto perché i file sono stati riordinati.

| Ambiente | Pin principali |
|---|---|
| `turbovla-libero` | torch 2.3.1/cu121, torchvision 0.18.1, transformers 4.56.0, TF 2.20, TFDS 4.9.3, NumPy 1.26.4, MuJoCo 2.3.7 |
| `turbovla-robotwin` | torch 2.6.0/cu124, torchvision 0.21.0, transformers 4.57.0, NumPy 1.26.4 |

Per completare FlashAttention RoboTwin usare, su un nodo/toolchain adatto:

```bash
sbatch --partition=gpu_a40 --gres=gpu:1 scripts/cluster/envs.sh \
  --benchmark robotwin --with-flash-attn
```

Senza il flag, la sua installazione viene esplicitamente differita. Il training
RoboTwin controlla la versione `2.7.4.post1`. L'eccezione preesistente al difetto
di metadata della wheel `pipablepytorch3d` resta limitata e accompagnata da un
test funzionale; non vengono ignorati genericamente errori di dipendenze.
I manifest Conda/pip sono in `manifests/environments/`.

Il simulatore LIBERO si installa **dopo** il download, con lo stesso file:

```bash
sbatch scripts/cluster/envs.sh --benchmark libero --simulator-only
```

Il simulatore **RoboTwin richiede un ambiente separato**, secondo la sua
[guida ufficiale](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
Il nostro ambiente `turbovla-robotwin` è quello della policy/training, non una
certificazione di installazione del simulatore. Sul cluster questo simulatore
non risulta ancora predisposto: non avviare la sua evaluation prima del setup.

## 3. Un solo downloader e verificatore

Accettare i termini del DINO necessario (ViT-B per LIBERO, ViT-L per RoboTwin)
su Hugging Face e autenticarsi con `hf auth login`. Il token rimane nella home
privata: gli script non spostano `HF_HOME` né registrano credenziali nei report.

```bash
bash scripts/cluster/assets.sh --benchmark libero --dry-run
sbatch scripts/cluster/assets.sh --benchmark libero
```

Per entrambi: `--benchmark all`. Per limitare il lavoro:

```bash
sbatch scripts/cluster/assets.sh --benchmark robotwin \
  --components models checkpoints
```

| Flag | Significato |
|---|---|
| `--benchmark libero/robotwin/all` | Seleziona solo gli asset necessari a quel benchmark |
| `--components ...` | Uno o più fra `models datasets checkpoints simulators`; default tutti |
| `--max-workers 8` | Parallelismo dei download HF |
| `--verify-only` | Non scarica payload; verifica ciò che esiste |
| `--online` | In verifica, consulta l'indice completo della revisione HF fissata |
| `--report /percorso/report.json` | Scrive esplicitamente un report, anche in modalità verifica |
| `--store /percorso/TurboVLA` | Root alternativa; altrimenti `TURBOVLA_STORE` o `$SCRATCH_FLASH/TurboVLA` |

Verifica completa contro l'indice pubblico, senza riscaricare i pesi:

```bash
sbatch scripts/cluster/assets.sh --benchmark libero --verify-only --online
```

Omettere `--online` per il controllo offline. Il download salva gli indici
remoti con revisione e hash. Per asset scaricati dai vecchi script, in assenza
di tali indici si usano i metadata HF già presenti. In quel caso controlliamo
anche tutti i 96 shard LIBERO attesi; per attestare la **completezza** di tutti
i file RoboTwin rispetto al repository remoto usare `--online`.

### Cosa viene scaricato e verificato

| Asset | Repository pubblico | Revisione fissata (abbreviata) |
|---|---|---|
| DINOv3 ViT-B | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719` |
| DINOv3 ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `ea8dc28` |
| BERT | `google-bert/bert-base-uncased` | `86b5e09` |
| GroundingDINO Swin-T OGC | `ShilongLiu/GroundingDINO` | `84311ae` |
| Dataset LIBERO | `openvla/modified_libero_rlds` | `a7c9ae1` |
| Dataset RoboTwin clean50 | `StarVLA/RoboTwin-Clean` | `070d3b8` |
| Checkpoint LIBERO unificato | `H-EmbodVis/TurboVLA` | `cb53005` |
| Checkpoint RoboTwin EMA 55k e metadata | `H-EmbodVis/TurboVLA` | `f7b0f53` |
| ZIP asset RoboTwin | `TianxingChen/RoboTwin2.0` | `9dc9299` |

Le revisioni complete sono nel catalogo in `_internal/assets.py`, non si segue un `main`
mobile. Il checkpoint RoboTwin fissato è quello ufficiale 55k già scaricato:
non serve duplicarlo nella cartella della nuova release LIBERO.

Il download verifica ogni file contro hash SHA256/LFS o Git-blob SHA1 HF;
alcuni pesi fondamentali hanno anche SHA256 fissati nel codice. Non sostituisce
silenziosamente file esistenti con contenuto diverso. I download mancanti
sono riprendibili. Le statistiche locali rilasciate vengono controllate.
La verifica dimostra identità con **questi asset pubblici**, non con copie
private degli autori né con un insieme di dati ricostruito autonomamente.

LIBERO usa quattro suite RLDS già `no_noops`: Spatial 432, Object 454, Goal 428,
Long 379 episodi da metadata, totale **1693**. Non rigeneriamo né convertiamo
in LeRobot questi dati. `regenerate_libero_no_noops.py` è per partire dagli
originali, rimuovere no-op/replay falliti e poi convertire in RLDS.
Manteniamo `libero_all4_stats.json`, chiave `libero_all4_no_noops`; non abbiamo
ricalcolato tutte le statistiche. `online_text_layout.json` descrive il padding,
non contiene embedding testuali precalcolati.

`--components simulators` scarica/verifica il checkout LIBERO fissato a
`8f1084e` e la configurazione locale; per RoboTwin scarica/verifica i tre ZIP.
Non installa driver, ambiente del simulatore RoboTwin o estrae automaticamente
decine di GB in un checkout non ancora concordato. Questi ZIP da soli **non**
significano che il simulatore sia pronto.

## 4. Evaluation: checkpoint sempre esplicito

```bash
sbatch --job-name=libero-checkpoint-paper scripts/cluster/evaluate.sh --benchmark libero \
  --checkpoint "$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth" \
  --suite all
```

Questo invia **un job da una GPU** ed esegue le quattro suite **in sequenza**.
Per una sola suite usare ad esempio `--suite libero_object`.
Per quattro job indipendenti in parallelo, senza array, inviare lo stesso file
quattro volte (scegliere un `run_label` nuovo a ogni benchmark, anche per i log):

```bash
checkpoint="$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth"
run_label="unified-eval-01"
result_root="$SCRATCH_FLASH/TurboVLA/results/libero/$run_label"
for suite in libero_spatial libero_object libero_goal libero_10; do
  sbatch --job-name="${run_label}-${suite}" scripts/cluster/evaluate.sh \
    --benchmark libero --checkpoint "$checkpoint" \
    --suite "$suite" --output "$result_root"
done
```

Le suite hanno sottocartelle distinte e non si sovrascrivono. Un fallimento
interrompe il job interessato; non vengono nascoste o rilanciate le suite fallite.

| Flag | Default LIBERO / significato |
|---|---|
| `--checkpoint PATH` | Obbligatorio, anche per checkpoint propri compatibili |
| `--trials` | 50 prove/task LIBERO; 100 RoboTwin |
| `--seed` | 7, solo LIBERO |
| `--chunk-size` / `--open-loop-steps` | 12 / 12, solo LIBERO |
| `--precision` | `bf16`; alternativa `fp32` |
| `--renderer` | `egl` sul cluster; alternativa `osmesa` se installato |
| `--task-ids 0,1` | Sottoinsieme LIBERO, non benchmark completo |
| `--save-video` | Attiva video LIBERO; default disattivati |
| `--stats-path PATH` | Statistiche per checkpoint custom; default quelle rilasciate |
| `--load-only` | LIBERO: caricamento rigoroso senza rollout |
| `--set KEY=VALUE` | Altri parametri LIBERO noti, es. `num_steps_wait=10`, `env_img_res=256` |
| `--output PATH` | Root dei risultati; default `results/BENCHMARK/NOME_JOB`; non sovrascrive suite già esistenti |
| `--skip-asset-check` | Salta intenzionalmente l'audit dei pretrained: non attestazione baseline |

La configurazione espansa viene salvata. Download automatici HF disabilitati.
Evaluation esegue **il simulatore**, non legge le dimostrazioni RLDS di training.
Per 40 task × 50 prove sono 2000 episodi. Cambiare parametri, task o numero
di prove produce un esperimento diverso dal benchmark completo.

Il formato del checkpoint deve essere compatibile con il modello/evaluator
scelto, non basta che abbia estensione `.pth`. In LIBERO sono preferiti e
richiesti i pesi `ema_model_state_dict`, salvo l'eccezione SHA256 per l'export
ufficiale di settembre, che li conserva sotto `model_state_dict`.
Per il confronto storico richiesto il 10 settembre sono ammessi anche i
quattro export della revisione HF `f7b0f53afa248408d20748f2579e446c7ce4119e`,
esclusivamente dopo verifica SHA256 contro l'indice LFS pubblico. Contengono
solo `model_state_dict`: li valutiamo come pubblicati, senza attribuire loro
provenienza EMA o convertire i tensori. Questa è un'eccezione locale di
compatibilità per la release storica, non la raccomandazione attuale degli autori.
La scelta EMA per checkpoint di training e il caricamento unified restano invariati.
Non convertiamo raw arbitrari in EMA, non adattiamo automaticamente architetture
incompatibili e non eliminiamo `strict=True`. Usare solo checkpoint fidati:
l'upstream carica archivi PyTorch tramite `torch.load`.

RoboTwin, soltanto dopo avere installato il simulatore separato:

```bash
sbatch scripts/cluster/evaluate.sh --benchmark robotwin \
  --checkpoint "$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors" \
  --robotwin-root "$SCRATCH_FLASH/TurboVLA/simulators/RoboTwin" \
  --robotwin-python /PERCORSO/ENV_SIMULATORE/bin/python \
  --trials 100
```

`--tasks adjust_bottle beat_block_hammer` limita le task; omesso = clean50.
Il wrapper usa il protocollo `scripts/robotwin/evaluate.sh`, BERT/DINO locali e
output di server e simulatore inviato direttamente agli stream Slurm, senza
duplicati `_server.log`/`_eval.log` (`ROBOTWIN_LOG_TO_STDIO=1`). Il comportamento
fuori dal launcher cluster resta quello upstream. Gli eventuali artefatti interni del simulatore
seguono anche la sua configurazione: mantenere il checkout nello scratch.
Una nuova architettura può richiedere anche codice e metadata compatibili:
non è un caricatore universale di checkpoint LeRobot/OpenVLA/π0.5.

## 5. Training: default visibili, modificabili

Prima visualizzare la configurazione senza avviare nulla:

```bash
bash scripts/cluster/train.sh --benchmark libero --dry-run
```

Comando completo, **da eseguire quando si decide di partire**:

```bash
sbatch --gres=gpu:4 --job-name=libero-baseline scripts/cluster/train.sh --benchmark libero
```

Le quattro GPU cooperano via DDP nello stesso modello mixed-suite.
Specificare `--gres=gpu:4` evita che una modifica dell'header Slurm (ad esempio
una prova a una GPU) cambi il batch globale della ricetta. Il launcher usa
il numero di GPU assegnate da Slurm, non forza quattro processi su una GPU.
Per LIBERO il comando generato usa ora esplicitamente `bin/torchrun` dello
stesso ambiente Conda dell'interprete selezionato, seguito da
`--standalone --nnodes=1 --nproc_per_node=N experiments/libero/train.py`
(il percorso reale del file è nella copia dei sorgenti del tentativo).
Prima usava `python -m torch.distributed.run`: è lo stesso entry point PyTorch.
`--standalone` configura il rendezvous locale per il nostro job a un nodo;
non è un iperparametro degli autori. RoboTwin mantiene il launcher Accelerate.
Il run viene creato in `$SCRATCH_FLASH/TurboVLA/training/libero/libero-baseline`.
`--output` permette una directory diversa, che deve essere nuova per un training
fresco: non crearla preventivamente. Se esiste già, il launcher si ferma.
Il training LIBERO parte da DINOv3 e BERT pretrained, più interazione/proiezione
testuale GroundingDINO; non dal TurboVLA ufficiale già allenato. Il precedente
test ha caricato 180 tensori di interazione e due di proiezione senza mismatch.
Il decoder delle azioni e i componenti nuovi partono dall'inizializzazione del
costruttore. DINO è allenabile, BERT congelato e online.

| Flag | Default LIBERO | Default RoboTwin |
|---|---:|---:|
| GPU (`sbatch --gres=gpu:N`, prima dello script) | 4 | 4 |
| `--batch-size` (per GPU) | 8 | 48 |
| `--grad-accum-steps` | 4 | 1 |
| `--max-steps` | 80000 | 55000 |
| `--warmup-steps` | 10000 | 1000 |
| `--lr` (gruppi allenabili) | 5e-5 | 5e-5 |
| `--save-steps` | 1000 | 5000 |
| `--seed` | 42 | 42 |
| `--num-workers` (per processo) | 4 | 8 |

LIBERO conserva policy FP32, DINO BF16 autocast, loss L1, AdamW beta
0.9/0.95, epsilon 1e-8, weight decay 1e-10 nei gruppi decay, clipping 1.0,
EMA 0.999, due viste 256px, sei layer di interazione e chunk 12.
`min_lr_ratio=1.0` mantiene il LR costante dopo warmup, non introduce un cosine
decay. La durata dello schedule LIBERO rimane 80000 anche con `--max-steps 3`.

Altri flag LIBERO si impostano con `--set`, usando i nomi del parser ufficiale:

```bash
bash scripts/cluster/train.sh --benchmark libero --dry-run \
  --batch-size 16 --set precision=fp32 --set dinov3_precision=bf16_autocast
```

Batch 16 × 4 × 4 = 256 è la variante del testo del paper, **non** la
raccomandazione aggiornata. Sono accettati anche parametri architetturali e
percorsi noti, es. `--set vla_feature_enhancer_layers=4`, `--set dataset_dirs=...`.
Per dati/pesi custom può servire `--skip-asset-check`; il run non è allora una
baseline verificata. Gli override sono registrati e i nomi sconosciuti rifiutati.

RoboTwin usa il suo YAML rilasciato; `--set` accetta chiavi YAML esistenti con
notazione puntata, es. `--set trainer.ema_decay=0.999`. I campi non sovrascritti
restano nel YAML ufficiale, copiato insieme al codice del run.

```bash
bash scripts/cluster/train.sh --benchmark robotwin --dry-run
```

W&B è disabilitato per default nel wrapper cluster; impostare `WANDB_MODE`
esplicitamente se si vuole usarlo. Questo non cambia la loss o l'optimizer.
La parte training/evaluation RoboTwin non è stata validata end-to-end qui.

### Ripresa e riproducibilità

**Questa struttura di archiviazione è nostra, non un output standard upstream.**
Gli script interni non sono soltanto costruttori di comandi: verificano gli
asset/ambienti, rifiutano sovrascritture e registrano configurazioni e sorgenti.

| Elemento LIBERO | Origine |
|---|---|
| File `turbovla_step_N.pth` | Salvataggio del trainer originale nella directory `--checkpoint_dir` |
| `training/libero/NOME_JOB/checkpoints/` | Percorso scelto dal nostro launcher e passato al trainer |
| `attempts/`, `resume_contract.json`, `.training.lock` | Gestione locale dei tentativi, controlli di ripresa e lock |
| `config.json`, `command.txt`, `execution.json`, `git-head.txt`, `pip-freeze.txt`, `source/` | Registrazioni e copia dei sorgenti aggiunte da noi |
| JSON di metriche della evaluation | Scritto dall'evaluator originale tramite `--result_json_path` |
| `results/libero/NOME_JOB/SUITE/`, `checkpoint.json` | Organizzazione per suite e identificazione del checkpoint aggiunte da noi |

Il trainer originale, lanciato direttamente, **non** crea `attempts/` o
`resume_contract.json`: scrive i checkpoint nella directory che gli passi.
L'evaluator originale valuta una suite per invocazione; il nostro `--suite all`
esegue quattro invocazioni sequenziali e le colloca in quattro sottocartelle.
I metadati non cambiano loss o rollout e non sono log duplicati, ma sono
aggiunte reali. I sorgenti eseguiti sono una copia del checkout locale, non un
download upstream pulito: includono il fix locale EMA descritto sotto e, per
evaluation, l'eccezione di caricamento dell'export ufficiale verificato via
SHA256 descritta nella sezione evaluation. Anche gli adattamenti RoboTwin
per interpreti e log sono modifiche locali, non parte della guida originale.

Ogni run ha directory nuova, configurazione, comando, versioni, hash degli
asset e copia del codice. La copia viene fatta **all'avvio del job**, non alla
sottomissione: non cambiare branch mentre job sono in coda se vuoi fissare
esattamente quella versione. Gli ambienti Conda e gli asset restano condivisi.

Ripresa LIBERO dopo verifica dell'ultimo checkpoint completo:

```bash
sbatch --job-name=libero-baseline-ripresa scripts/cluster/train.sh --benchmark libero --resume \
  --output "$SCRATCH_FLASH/TurboVLA/training/libero/libero-baseline"
```

Rifiuta modifiche a ricetta, GPU, codice modello, asset e versioni principali.
Il solo limite `max_steps` può cambiare, mantenendo lo schedule configurato.
Ripristina pesi raw, optimizer, scheduler e **EMA**. Il fix locale EMA non altera
il training fresco; RNG e posizione del lettore dati non sono ripristinati,
quindi non è una continuazione identica bit per bit.
Il salvataggio upstream non è atomico: un time limit durante la scrittura può
lasciare un checkpoint incompleto. Non lo cancelliamo/saltiamo automaticamente.
La ripresa RoboTwin non è esposta perché non validata.

I tentativi sono `attempts/initial/`, poi `attempts/resume-01/`, `resume-02/`,
ecc.: il numero è l'ordine delle riprese, non un ID casuale. Ogni tentativo salva
configurazione, comando, versioni e sorgenti, **non una seconda copia dei log**.
`execution.json` conserva job ID e data come metadati per rintracciare il job
nello scheduler. I run già presenti con i vecchi nomi restano invariati: per
riprenderli indicare il loro percorso originale con `--output`.
Per una ripresa usa un nome job diverso ma lo stesso `--output`: così mantieni
i checkpoint nella directory originale e non sovrascrivi i log del primo avvio.

Non scegliere il checkpoint solo dalla loss. Gli autori confrontano success
rate in simulazione; il nuovo export LIBERO è dichiarato EMA a 34k, non a 80k.
Valutare lo **stesso checkpoint** sulle quattro suite e dichiarare lo step.
Il precedente test a tre aggiornamenti verificava il funzionamento, non il successo finale.

## 6. Slurm e output

| Risorsa Slurm | `envs.sh` | `assets.sh` | `evaluate.sh` | `train.sh` |
|---|---|---|---|---|
| `--partition` | cpu_sapphire | cpu_sapphire | gpu_a40 | gpu_a40 |
| `--gres` | nessuna GPU | nessuna GPU | gpu:1 | gpu:4 |
| `--cpus-per-task` | 8 | 8 | 8 | 32 |
| `--mem` | 16G | 16G | 64G | 256G |
| `--time` | 04:00:00 | 1-00:00:00 | 6:50:00 | 23:50:00 |

Queste opzioni vanno **prima del nome del file**, i flag dell'esperimento dopo:

```bash
sbatch --partition=gpu_a40 --gres=gpu:4 --cpus-per-task=32 \
  --mem=256G --time=23:50:00 --job-name=libero-baseline scripts/cluster/train.sh \
  --benchmark libero --batch-size 8 --grad-accum-steps 4
```

Il training legge il numero di GPU da `SLURM_GPUS_ON_NODE`; non passare
`--gpus` allo script. Cambiare le GPU senza adeguare batch/accumulo cambia il
batch globale. Questi launcher sono **single-node, una task Slurm**, con DDP
che crea i processi locali: non aumentare `--ntasks`/`--nodes`.
La evaluation LIBERO richiede una sola GPU per job.

Ci sono **soltanto i log Slurm**, separati per operazione:

```text
/home/gpepe/ws/logs/
├── smolvla/                         file precedenti non-TurboVLA, spostati qui
└── turbovla/
    ├── training/<nome-job>.out      stdout del training
    ├── training/<nome-job>.err      stderr del training
    ├── evaluation/<nome-job>.out    stdout della evaluation
    ├── evaluation/<nome-job>.err    stderr della evaluation
    ├── assets/<nome-job>.out/.err   download/verifica
    └── envs/<nome-job>.out/.err     installazione ambienti
```

`%x` viene sostituito dal nome del job; non usiamo più `%j`. In LIBERO
`--log_path` è vuoto: l'evaluator conserva il logger su console senza creare
`evaluation.log`. Il training eredita stdout/stderr senza tee o `train.log`.
I diagnostici Xet usano la destinazione console (`HF_XET_LOG_DEST` vuota).
Non vengono più creati `logs.json` o cartelle applicative con hash/timestamp.

I launcher non impostano `--open-mode` e non usano più l'helper `slurm.sh`.
L'opzione append è supportata da Slurm, ma è stata rimossa su richiesta.
La configurazione del cluster verificata ha `JobFileAppend=0`: riusare un nome
job può troncare i suoi `.out/.err`, anche se il programma poi rifiuta un run
già esistente. Scegli **nomi job distinti a ogni sottomissione**, incluse le
riprese; `--output` consente di continuare a usare la stessa directory del run.
Non esiste più il lock sui nomi dei log: non lanciare due job dello stesso tipo
con lo stesso nome. Rimangono i controlli del downloader e del training sulle
rispettive directory dati/run, che sono una protezione diversa.

Report JSON, configurazioni, video e checkpoint restano nello scratch.
I log interni di servizi opzionali riattivati esplicitamente (ad esempio W&B)
seguono la loro configurazione; il launcher mantiene W&B disabilitato di default.
Le richieste Slurm sono risorse del cluster, non impostazioni scientifiche degli autori.
Nessun job viene rilanciato automaticamente dopo timeout/fallimento.

Pulizia del 7 settembre 2026, su richiesta e senza job attivi:

- I checkpoint smoke (circa 4 GiB) erano stati eliminati nella prima pulizia.
- Nella pulizia completa dei log sono stati eliminati i 42 file TurboVLA rimasti
  nella root `logs/` e il precedente contenuto di `logs/turbovla/`, inclusi i
  duplicati applicativi, i vecchi log Xet e l'archivio dei metadati smoke.
  Non è stato conservato un backup di questa eliminazione.
- I 189 file restanti della root `logs/` sono stati spostati in `logs/smolvla/`,
  senza cancellarli o rinominarli. I launcher degli altri progetti non sono stati modificati.
- Nessun dataset, modello, checkpoint di training valido, simulatore o risultato
  JSON delle evaluation nello scratch è stato cancellato o rinominato in questa
  pulizia dei log. Eventuali log storici accanto ai risultati nello scratch
  non rientravano nella pulizia di `ws/logs/`.

Nessuna pulizia automatica dei checkpoint di training: la selezione resta
esplicita, per non perdere run o punti di ripresa validi.

Ordine per una nuova installazione: attendere il completamento di `envs.sh`,
poi `assets.sh`, poi `envs.sh --simulator-only` per LIBERO, infine evaluation
o training. Non sottomettere installazioni e training contemporaneamente sugli
stessi ambienti. Per anteprime senza coda usare `bash FILE.sh ... --dry-run`;
`sbatch FILE.sh ... --dry-run` prenota comunque un job, anche se non esegue il carico.

```text
$SCRATCH_FLASH/TurboVLA/
├── models/
│   ├── dinov3-vitb16/                LIBERO pretrained
│   ├── dinov3-vitl16/                RoboTwin pretrained
│   ├── bert-base-uncased/
│   └── groundingdino/
├── pretrained/
│   ├── TurboVLA-unified-cb53005/     LIBERO ufficiale settembre
│   └── TurboVLA/                     release storica; anche RoboTwin EMA 55k
├── datasets/
│   ├── libero/                      quattro *_no_noops/1.0.0 in RLDS
│   └── robotwin/Clean/               50 dataset LeRobot
├── simulators/LIBERO/                checkout esterno fissato
├── simulator_assets/robotwin/        ZIP, non simulatore installato
├── config/libero/config.yaml         percorsi per il simulatore
├── cache/huggingface/                cache dati; non il token
├── manifests/                       audit, indici HF, versioni degli env
├── training/
│   ├── libero/NOME_JOB/              nuovi checkpoint, attempts e sorgenti
│   └── robotwin/NOME_JOB/
└── results/
    ├── libero/NOME_JOB/SUITE/         results.json, execution.json, config e sorgenti
    └── robotwin/NOME_JOB/clean50/
```

Il fallback di root per questo account è `/mnt/beegfs/gpepe/TurboVLA` se
`SCRATCH_FLASH` non è definito; `--store`/`TURBOVLA_STORE` lo sovrascrivono.
Conservare 80 checkpoint LIBERO può richiedere circa **160 GiB**, sulla base
del test reale. Nessuna retention/cancellazione automatica è abilitata.

## 7. Stato delle verifiche

Aggiornamento del **10 settembre 2026**, dopo la conclusione dei job:

- **38 test passati**, rieseguiti con `python -B -m unittest discover -s tests -v`:
  CLI/launcher, default upstream, hash sintetici, export ufficiali e continuità EMA.
  Copertura e limiti sono nel [README dei test](../../tests/README.md).
- Il training LIBERO reale `libero-baseline`, job `1923969`, ha completato
  **80k step su quattro GPU**, batch globale 128. I suoi 80 checkpoint restano
  nello scratch. Questa verifica sostituisce il vecchio stato «DDP da validare».
- Complete le evaluation dei nostri checkpoint 34k/80k e delle due release
  ufficiali: [tabella, protocollo e warning](LIBERO_RESULTS.md). Tutti gli 8.000
  episodi registrati nei log coincidono con i 16 JSON, anche task per task.
- Audit reale degli hash di tutti i **10.531 file** selezionati nei nove gruppi
  del catalogo, incluso l'intero dataset RoboTwin selezionato. Anche gli indici
  remoti delle revisioni HF fissate coincidono con quelli locali: non solo
  presenza dei file, ma completezza rispetto ai pattern del catalogo.
  Ricalcolati separatamente anche gli SHA256 dei quattro export LIBERO storici.
- Checkout/configurazione LIBERO e hash delle statistiche/layout rilasciati
  coerenti; nessun collegamento simbolico rotto rilevato nello scratch.
- La pipeline **RoboTwin completa resta da validare**: gli ZIP non equivalgono
  al checkout/ambiente del simulatore installato. Non è stato lanciato RoboTwin.
  Nell'ambiente policy manca ancora `flash-attn`, richiesto dal training.
- Pin degli ambienti verificati; `pip check` LIBERO passa. RoboTwin segnala
  soltanto il già documentato difetto della wheel `pipablepytorch3d 0.7.6`:
  rieseguito con successo il test CPU `quaternion_to_matrix`. Questo test
  limitato non certifica tutte le operazioni PyTorch3D o la pipeline GPU.

Non sono stati rilanciati training/evaluation, reinstallati ambienti o scaricati
payload durante questo audit. La precedente prova `sbatch --test-only` dei quattro
launcher rimane una verifica storica delle risorse, non un nuovo test del controller.
I test CPU sul login node non certificano CUDA o i mount su tutti i nodi.

### Diagnosi di un job che scompare senza log

Le opzioni Slurm devono precedere il file, quelle applicative devono seguirlo:

```bash
sbatch --job-name=libero-baseline scripts/cluster/train.sh --benchmark libero
```

Nel job `1923940` del 7 settembre 2026, `--job-name=libero-baseline` era stato
inserito dopo il file: il nome effettivo era quindi `turbovla-train-baseline-01`.
Questo errore di posizione è distinto dal fallimento osservato dal controller:
`FAILED`, `Reason=JobLaunchFailure`, `ExitCode=0:53`, durata zero, nodo
`compute-4-13`. Slurm aveva assegnato quattro GPU, 32 CPU e 256 GiB di memoria;
non sono comparsi i file `.out/.err`. La richiesta `--gres=gpu:4` è supportata
dai nodi della partizione, ma ciò non certifica il corretto avvio sul nodo.
Non ci sono evidenze di un errore Python o di memoria CUDA in questo job.

**Causa individuata con job diagnostici il 7 settembre 2026:** su
`compute-4-13` manca il mount NFS di `/home` e `/home/gpepe` non esiste.
Di conseguenza non sono accessibili repository, interprete Conda e directory
dei log. Lo scratch BeeGFS è invece visibile. Il training breve `1923959`
ha riprodotto il fallimento prima di Python; le prove CPU `srun` `1923960`
e `1923961`, avviate da `/tmp` con output al terminale, hanno mostrato
direttamente i percorsi mancanti e l'assenza del mount.

Il confronto `1923965` su `compute-3-14` ha trovato `/home` montata da
`192.168.122.40:/hpc`, repository e Conda accessibili e directory log scrivibile.
La prova batch CPU `1923967` ha creato `libero-diagnosi-log.out/.err` ed
eseguito il dry-run del launcher: non è una verifica di CUDA o del training DDP.

La successiva prova GPU `1923968` (`libero-diagnosi-gpu`) su `compute-4-16`
ha completato tre aggiornamenti reali: loss `0.41180`, `0.39705`, `0.37634`.
Ha usato il launcher corrente, l'audit degli asset abilitato e le quattro suite
RLDS, con una GPU, batch per GPU 8 e accumulo 16 (batch globale 128).
Override esclusivamente diagnostici: `--max-steps 3 --grad-accum-steps 16
--set save_final=false --set log_freq=1`; risorse 8 CPU, 64 GiB, 15 minuti.
Il limite dello schedule è rimasto 80k. Non sono stati richiesti checkpoint di
test; erano stati conservati log Slurm e metadati/sorgenti separati. I metadati
del run nello scratch sono stati eliminati nella pulizia del 10 settembre (§9).
Questa prova verifica il training su una GPU, **non** il DDP a quattro GPU
né la riproduzione dei risultati del paper. Le quattro GPU su un nodo diverso
da `compute-4-13` non erano subito disponibili, quindi non è stato avviato
un training completo da 80k step durante la diagnosi.

Workaround temporaneo, senza cambiare gli iperparametri del training:

```bash
sbatch --exclude=compute-4-13 --gres=gpu:4 --job-name=libero-baseline \
  scripts/cluster/train.sh --benchmark libero
```

L'esclusione può allungare la coda: `PENDING (Resources)` è diverso dal
fallimento immediato. Chiedere agli amministratori di ripristinare il mount
`/home` sul nodo; non sono stati modificati mount o configurazione Slurm.
Non abbiamo aggiunto l'esclusione permanentemente agli script: è un'opzione
di sottomissione temporanea da rimuovere quando il nodo viene riparato.
I precedenti job utente `1923953` e `1923956`, con lo stesso nome e gli stessi
log, sono stati annullati con consenso esplicito prima delle nuove prove.

`sacct` da `login1` non raggiunge il servizio configurato su `localhost:6819`.
Il controller conserva i job terminati solo brevemente (`MinJobAge=300`),
quindi un ID non più visibile non significa che non sia mai stato sottomesso.
Cambiare `torchrun`, append o il numero di GPU non ripristina un mount mancante.

## 8. Nuovo confronto delle release ufficiali (10 settembre 2026)

Job completati, una GPU A40 ciascuno, `--exclude=compute-4-13`, limite 20h50.
Risultati nel [confronto LIBERO](LIBERO_RESULTS.md):

| Nome Slurm / directory risultati | Job ID | Pesi |
|---|---|---|
| `libero-author-chekpt-unified` | `1929454` | Unico `TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth` per tutte le suite |
| `libero-author-chekpt-non-unified` | `1929455` | Quattro export storici da `TurboVLA/checkpoints/libero/` |

La mappatura della release storica è `libero_spatial → spatial.pth` (51k),
`libero_object → object.pth` (75k), `libero_goal → goal.pth` (49k),
`libero_10 → long.pth` (60k). Gli autori chiariscono che provengono dal
training mixed-suite, non da quattro training indipendenti.

Entrambi i job usano 50 prove/task, seed 7, chunk/open-loop 12, BF16, EGL,
wait 10, immagini 256px, controllo relativo e statistiche `libero_all4_no_noops`.
Le quattro suite sono eseguite in quattro processi sequenziali sulla GPU
assegnata al job, per 2000 episodi totali. Il job unified usa
`scripts/cluster/evaluate.sh --suite all`; quello storico usa un ciclo POSIX
in `sbatch --wrap` che richiama **lo stesso** `evaluate.sh` una volta per suite
con il checkpoint corrispondente. Nessun nuovo launcher permanente.

I risultati sono `results/libero/NOME_JOB/SUITE/` nello scratch; i soli log
sono `ws/logs/turbovla/evaluation/NOME_JOB.out/.err`. Non modificare i sorgenti
mentre le suite sequenziali sono in corso: ogni invocazione ne salva una copia.
Il protocollo è lo stesso delle evaluation `libero-baseline-34k` e `-80k`;
l'eccezione SHA256 aggiunta per gli export storici non cambia il percorso EMA
usato dai checkpoint di training, né quello dell'export unified.

La guida upstream prescrive una suite per invocazione e lascia lo scheduling
multi-GPU all'utente: quattro job separati oppure quattro invocazioni
sequenziali non sono due protocolli scientifici diversi. Non è dimostrato
che una particolare collocazione sulle GPU garantisca i numeri del paper.
La raccomandazione corrente è usare **il solo checkpoint unified su tutte
le suite**; la release storica è mantenuta esclusivamente come confronto.

Pulizia esplicitamente richiesta, completata senza job attivi: eliminate
senza backup le directory scratch `results/libero_official_eval`,
`results/libero_official_eval_isolated`, `results/libero_unified_eval` e
`results/libero/turbovla_libero-20260907T021005.521770Z` (circa 1,3 MiB).
La directory documentale `scripts/cluster/archive` è stata rimossa perché non
più necessaria. **Preservati** i risultati 34k/80k, tutti i checkpoint, i
dataset e gli ambienti. I vecchi log Slurm delle release
ufficiali erano già assenti dalla directory log attiva.

Riferimenti: [guida evaluation](https://github.com/H-EmbodVis/TurboVLA/blob/main/experiments/libero/README.md#evaluation),
[release unified](https://github.com/H-EmbodVis/TurboVLA/issues/11#issuecomment-5504905902),
[EMA e selezione del checkpoint](https://github.com/H-EmbodVis/TurboVLA/issues/12#issuecomment-5504917736).

## 9. Audit e pulizia dello scratch — 10 settembre 2026

Con coda utente vuota, eliminate su richiesta **senza backup**:

- `training/libero/20260907T025108.035301Z`: circa **160 GiB**, inclusi 80
  checkpoint. Era il vecchio training completo a una GPU/batch globale 32,
  non un semplice smoke test e non la baseline a quattro GPU ora confrontata.
- `training/libero/libero-diagnosi-gpu`: circa 728 KiB, prova diagnostica di
  tre step senza checkpoint. Il suo esito resta descritto nella sezione 7.
- `MUJOCO_LOG.TXT` nella root della repository: vecchi warning del 4–5 settembre.
  Non è un input del modello o del simulatore. MuJoCo può ricrearlo se emette
  nuovi warning; aggiunto a `.gitignore`, senza sopprimere i warning stessi.

Preservati tutti gli 80 checkpoint di `training/libero/libero-baseline`, le
quattro evaluation complete (16 suite), dataset, modelli, release ufficiali,
configurazioni, manifest e installazioni. Conservati anche i quattro registri
MuJoCo nelle copie `results/libero/NOME_JOB/libero_10/source/`: appartengono
alle evaluation appena concluse e documentano il warning descritto nel report.

Ingombro dopo la pulizia, arrotondato da `du -h`: **196 GiB** complessivi.

| Area | Ingombro circa | Perché resta |
|---|---:|---|
| `training/libero/libero-baseline/` | 160 GiB | 80 checkpoint, contratto di ripresa e sorgenti/config del run reale |
| `results/libero/` | 11 MiB | Quattro confronti completi con provenienza; non sono log duplicati |
| `datasets/libero/` | 9,6 GiB | Quattro dataset RLDS `no_noops`, 1.693 episodi di training |
| `datasets/robotwin/` | 3,8 GiB | Dataset clean50 LeRobot verificato |
| `models/` | 2,6 GiB | DINOv3 B/L, BERT, GroundingDINO |
| `pretrained/` | 4,9 GiB | Unified, quattro export storici LIBERO e checkpoint RoboTwin |
| `simulators/LIBERO/` | 740 MiB | Checkout e asset del simulatore utilizzato |
| `simulator_assets/robotwin/` | 14 GiB | ZIP ufficiali per futura installazione RoboTwin |

`datasets/libero_raw/` è vuota ma resta il percorso configurato per eventuali
dimostrazioni originali HDF5 del simulatore. Non è il dataset RLDS usato dal
training e non viene letto dall'evaluation. Non sostituirla con i TFRecord.
Le cache metadata `.cache/huggingface/download/` dentro gli asset servono a
verificare revisioni/hash e riprendere download: non sono scarti da cancellare.
La presenza dei file `.download_assets.lock`/`.training.lock` non indica da sola
un processo bloccato: il lock effettivo è mantenuto dal sistema durante il processo.

Correzioni non scientifiche di questo audit: messaggi single-node corretti nei
launcher assets/evaluation; errore esplicito se `--job-name`/`-J` è dopo il nome
di quei launcher o di `train.sh`; esempio parallelo con nomi dei log distinti
per run; documentazione aggiornata. Nessun cambiamento ai parametri, ai pesi,
all'architettura, ai risultati o alle copie dei sorgenti dei run completati.
L'audit non ricalcola da zero le statistiche e non rilegge tutti i 160 GiB dei
checkpoint della baseline: ne verifica l'inventario, non ogni tensore di ogni step.

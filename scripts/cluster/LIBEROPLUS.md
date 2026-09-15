# LIBERO+ con TurboVLA — solo evaluation

Ricerca e integrazione del 10 settembre 2026. LIBERO standard e i risultati
già validati restano separati; non cambiamo pesi, architettura o training.

## Che cosa misura

LIBERO+ misura la robustezza di policy già addestrate. Le varianti modificano
l'ambiente o l'osservazione mantenendo l'interfaccia LIBERO. L'esperimento che
vogliamo fare è **TurboVLA addestrata su LIBERO, valutata su LIBERO+ senza
fine-tuning**. Un calo rispetto al 97,30% su LIBERO non implica automaticamente
un errore di installazione: il benchmark valuta condizioni perturbate.
Riferimenti: [paper](https://arxiv.org/abs/2510.13626),
[repository ufficiale](https://github.com/sylvestf/LIBERO-plus).

Il codice pubblico contiene queste quattro suite:

| Suite da passare all'evaluator | Varianti / episodi con una prova |
|---|---:|
| `libero_spatial` | 2.402 |
| `libero_object` | 2.518 |
| `libero_goal` | 2.591 |
| `libero_10` | 2.519 |
| Totale | **10.030** |

Non si chiamano `libero_plus_spatial`, ecc. nella repository ufficiale usata qui.
`libero_90`, sebbene presente nel codice, non fa parte di questo totale.
La [guida ufficiale](https://github.com/sylvestf/LIBERO-plus#-evaluation)
prescrive **`num_trials_per_task=1`**, perché le perturbazioni sono già task
distinti; non sono i 40 task standard da ripetere una sola volta.

Conteggi verificati direttamente nel task map e in `task_classification.json`:

| Categoria | Varianti |
|---|---:|
| Camera Viewpoints | 1.599 |
| Robot Initial States | 1.550 |
| Language Instructions | 1.537 |
| Light Conditions | 1.142 |
| Background Textures | 1.076 |
| Sensor Noise | 1.601 |
| Objects Layout | 1.525 |

Per confrontare la robustezza conserveremo risultati per suite e per categoria.
I JSON dell'evaluator contengono gli esiti per task; l'aggregazione per categoria
va ricavata dalla classificazione ufficiale, non dal nome della suite.
I suoi ID sono **1-based**, mentre `--task-ids` e i JSON TurboVLA sono **0-based**:
il task 0 corrisponde all'elemento con `id=1`. Il totale si calcola come
successi/10.030, non facendo la media non pesata delle quattro suite, che hanno
numerosità diverse. Un sottoinsieme diagnostico non è un risultato completo.

## Cosa scarichiamo e cosa riusiamo

Nuovi file, sempre nello scratch:

- Codice ufficiale `sylvestf/LIBERO-plus`, commit
  `4976dc30028e805ff8094b55501d532c48fec182`; checkout sparse del codice LIBERO,
  senza i video del sito. Include definizioni task BDDL, stati iniziali e classificazione.
- `assets.zip` da [Sylvest/LIBERO-plus](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/dd2bd61b7d9a6fef1abc52d606e983b41886a149),
  revisione `dd2bd61b7d9a6fef1abc52d606e983b41886a149`:
  **6.395.849.578 byte** compressi, circa 6,40 GB / 5,96 GiB.
  SHA256 `96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf`.
  Contiene geometrie, oggetti, texture e scene; non dimostrazioni di training.
  Dopo verifica viene estratto in `LIBERO-plus/libero/libero/assets/`.
  Lo ZIP fissato contiene il prefisso di build degli autori
  `inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/`:
  il nostro estrattore rimuove esclusivamente questo prefisso, mantenendo `assets/`.
  Sono 448.799 file, 8.953.192.654 byte non compressi; l'estrazione può richiedere
  tempo per l'elevato numero di file. Non viene creata una directory `inspire/`.

Riusiamo **gli stessi** DINOv3 ViT-B, BERT, statistiche
`libero_all4_stats.json`/`libero_all4_no_noops` e checkpoint TurboVLA già presenti.
Per il primo confronto è consigliato l'unified ufficiale; successivamente si
possono usare i nostri checkpoint EMA 34k/80k senza modificare il modello.

**Non scarichiamo** RLDS, dataset LeRobot LIBERO+, dimostrazioni HDF5 o pesi
OpenVLA-OFT+: questi ultimi servono a esperimenti di altre policy/fine-tuning,
non a valutare TurboVLA. Non rigeneriamo `no_noops` e non ricalcoliamo statistiche
su LIBERO+: l'evaluation deve conservare la normalizzazione della policy addestrata.

## Ambiente isolato

LIBERO e LIBERO+ installano entrambi il pacchetto Python `libero`.
Per questo `turbovla-liberoplus` è un **clone Conda separato** di
`turbovla-libero`: preserva Torch 2.3.1, Transformers 4.56.0, NumPy 1.26.4,
MuJoCo 2.3.7 e le altre dipendenze già validate. Solo nel nuovo ambiente
sostituiamo la distribuzione editable LIBERO con LIBERO+.

Il launcher controlla anche la coerenza di `pip` nel clone. Se Conda ricopia
file di una versione aggiornata via pip insieme a quella più vecchia dei propri
metadati, può risultare un'installazione mista non avviabile. In quel caso usa
il pip funzionante dell'ambiente sorgente con l'opzione ufficiale
[`--python`](https://pip.pypa.io/en/stable/topics/python-option/) indirizzata
esclusivamente al clone: scarica prima la wheel della stessa versione sorgente,
rimuove tramite pip i record sovrapposti nel clone e installa quella sola wheel
senza dipendenze. Non aggiorna il sorgente, non ricrea l'ambiente e non modifica
Torch/Transformers. Se pip è già coerente, non lo reinstalla. Questo è un
accorgimento locale per Conda, non una modifica al protocollo LIBERO+.

La [procedura upstream](https://github.com/sylvestf/LIBERO-plus#-installation)
indica di installare il nuovo pacchetto e le dipendenze aggiuntive sopra un
ambiente LIBERO funzionante. Non installiamo il vecchio `requirements.txt`
integrale, che chiederebbe fra l'altro Transformers 4.21.1 e NumPy 1.22.4,
incompatibili con lo stack TurboVLA validato.

`extra_requirements.txt` richiede Wand e scikit-image senza versioni:
usiamo **nostri pin compatibili** `wand==0.6.13`, `scikit-image==0.24.0`, non
versioni dichiarate dal paper. Le librerie native ImageMagick/MagickWand,
expat e fontconfig vengono controllate sul nodo del job. Non eseguiamo `sudo`
o `apt` sul cluster: se una libreria manca, il job si ferma esplicitamente.

Viene aggiunto soltanto un marker `libero/__init__.py` al checkout esterno
per rendere affidabile l'installazione editable: problema descritto nella
[PR upstream #54](https://github.com/sylvestf/LIBERO-plus/pull/54).
Non modifichiamo perturbazioni, prompt, stati iniziali o logica del benchmark.

## Preparazione tramite Slurm

La preparazione attuale è completata: **non occorre rilanciare questi comandi**.
Sono il riferimento per una nuova installazione, che richiede prima l'ambiente
LIBERO sorgente funzionante. In `assets.sh` e `envs.sh`, `all` mantiene il
significato precedente, LIBERO + RoboTwin: LIBERO+ va richiesto esplicitamente.

```bash
cd /home/gpepe/ws/TurboVLA
bash scripts/cluster/assets.sh --benchmark liberoplus --dry-run
bash scripts/cluster/envs.sh --benchmark liberoplus --dry-run

sbatch --job-name=liberoplus-assets scripts/cluster/assets.sh --benchmark liberoplus
# Attendere che asset, estrazione e configurazione siano completati:
sbatch --job-name=liberoplus-env scripts/cluster/envs.sh --benchmark liberoplus
```

In alternativa, sottomettere il secondo con `--dependency=afterok:ID_DOWNLOAD`
e `--kill-on-invalid-dep=yes`, prima del nome del file. I log sono in
`/home/gpepe/ws/logs/turbovla/assets/` e `envs/`, con il nome del job.
Non occorre una GPU per questi due job.

Verifica successiva, senza download:

```bash
sbatch --job-name=liberoplus-verifica scripts/cluster/assets.sh \
  --benchmark liberoplus --verify-only --online
```

Si verificano hash dell'archivio e degli asset HF selezionati, revisione Git,
10.030 task/classificazioni, BDDL risolti, presenza degli stati iniziali e
completezza per dimensione dei file estratti. Il manifest deve avere esattamente
448.799 file e 8.953.192.654 byte complessivi, con percorsi normalizzati sotto
`assets/`: un manifest vuoto o parziale non passa. Il CRC è controllato durante
l'estrazione; il preflight non ricalcola ogni hash dei singoli file estratti.
Un'estrazione interrotta senza marker richiede ispezione prima di riprovare:
non sovrascriviamo automaticamente una directory preesistente.
Clone, estrazione e configurazione sono protetti da `.prepare_liberoplus.lock`,
oltre al lock del download HF: due job concorrenti non possono prepararli insieme.
I controlli in sola lettura non creano lock. Il preflight dell'ambiente non
aggiunge il checkout del simulatore al `PYTHONPATH`, così verifica la vera
installazione editable.

## Evaluation TurboVLA

Dopo la preparazione, anteprima senza GPU:

```bash
bash scripts/cluster/evaluate.sh --benchmark liberoplus \
  --checkpoint "$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth" \
  --suite libero_spatial --dry-run
```

Per una prima **prova diagnostica**, non un punteggio del benchmark:

```bash
sbatch --job-name=liberoplus-unified-check --time=00:30:00 \
  scripts/cluster/evaluate.sh --benchmark liberoplus \
  --checkpoint "$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth" \
  --suite libero_spatial --task-ids 0
```

Dopo aver verificato anche task camera/robot/language/noise, il confronto
completo può essere lanciato una suite per job (una GPU per job, senza array):

```bash
run_label="liberoplus-unified-01"
checkpoint="$SCRATCH_FLASH/TurboVLA/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth"
for suite in libero_spatial libero_object libero_goal libero_10; do
  sbatch --job-name="${run_label}-${suite}" --time=20:50:00 \
    scripts/cluster/evaluate.sh --benchmark liberoplus \
    --checkpoint "$checkpoint" --suite "$suite" \
    --output "$SCRATCH_FLASH/TurboVLA/results/liberoplus/$run_label"
done
```

Il primo run completo è terminato per timeout dopo oltre 23 ore di evaluation,
dopo Spatial e Object e durante Goal:
20h50 non sono quindi sufficienti per tutte le suite insieme su quel run.
Se serve, divideremo gli ID in sottoinsiemi disgiunti, senza saltare task falliti.
`--suite all` funziona in sequenza ma concentra 10.030 episodi in un solo job.
L'evaluator originale scrive il risultato alla fine della suite, non offre
qui una ripresa automatica dopo timeout: prima del benchmark lungo va stimata
la durata con un campione rappresentativo.

Il launcher richiama ancora `experiments/libero/evaluate.py`, con il nuovo
`--libero_root` e `LIBERO_CONFIG_PATH`. I default restano seed 7, chunk/open-loop
12, BF16, EGL, 256 px, attesa 10, controllo relativo, niente video. L'unico
default del protocollo cambiato è **una prova per variante**. Puoi usare
`--trials`, `--task-ids`, `--set`, `--checkpoint` e `--output` come per LIBERO.
Non esiste `train.sh --benchmark liberoplus` in questa integrazione.

Ogni suite usa lo stesso layout di risultati del launcher LIBERO, con in più
`benchmark.json`: registra revisione LIBERO+, conteggi, convenzione degli ID e
lunghezza testo. È un nostro metadato di riproducibilità, non un output upstream.
`results.json` contiene gli esiti prodotti dall'evaluator; stdout/stderr rimangono
nei soli log Slurm. Non viene prodotta automaticamente una tabella per categoria.

## Punti da non nascondere nel confronto

1. **Lunghezza testo TurboVLA:** l'unified contiene una `model_config` che il
   loader usa con priorità sui flag architetturali: `padding_length=21`, incluse
   le posizioni speciali BERT, e una mappa di lunghezze 11/14/21 per le istruzioni
   originali. Le nuove istruzioni non presenti nella mappa usano 21; se più lunghe
   vengono troncate. Non basta cambiare `max_text_len` né aggiungere
   `--set text_padding_length=64`: con questo checkpoint il flag non sovrascrive
   la configurazione incorporata. Una variante a testo più lungo richiede un
   override effettivo ed esplicito della configurazione e della mappa, da
   implementare e validare separatamente. I run attuali mantengono il checkpoint
   invariato; il valore CLI registrato non sostituisce la sua configurazione.
2. **Nomi BDDL virtuali:** l'[issue #60](https://github.com/sylvestf/LIBERO-plus/issues/60)
   segnala 6.287 file mancanti, ma nel codice verificato `ControlEnv` interpreta
   i suffissi `_view_..._initstate_...` e carica il BDDL base. Il nostro audit
   risolve questi nomi come l'upstream: nessun BDDL base o file di stato iniziale
   mancante per le 10.030 varianti. Non inventiamo file sostitutivi.
3. **Fog e risoluzione:** l'[issue #58](https://github.com/sylvestf/LIBERO-plus/issues/58)
   riguarda immagini più grandi della mappa fog 256×256. Manteniamo gli input
   TurboVLA a 256×256 e verifichiamo fog/motion blur a tale dimensione. Non
   cambiamo la generazione del rumore per uniformarla a pipeline di altre VLA.
4. **Prompt problematici:** l'[issue #59](https://github.com/sylvestf/LIBERO-plus/issues/59)
   segnala preamboli di riscrittura in alcuni BDDL. Non li correggiamo o
   escludiamo di nascosto. La versione del benchmark va sempre registrata.
5. **Riproducibilità altrui:** l'[issue #61](https://github.com/sylvestf/LIBERO-plus/issues/61)
   segnala discrepanze nella categoria Noise di π0. È una segnalazione di un
   utente su un'altra policy, non la dimostrazione di un bug TurboVLA né una
   ragione per alterare la nostra evaluation.
6. **Motion blur su immagini grayscale:** nel controllo locale del 10 settembre,
   l'immagine uniforme `(128,128,128)` viene ricodificata da ImageMagick come
   grayscale. `motion_blur` upstream gestisce quel caso solo a 224×224 e a
   256×256 restituisce erroneamente `(256,3)`. Confermato sul codice fissato:
   lo stesso controllo con un gradiente RGB a colori restituisce `(256,256,3)`.
   Il nostro preflight usa ora questo gradiente e controlla fog/motion blur alle
   severità 1 e 10, mantenendo i controlli su forma, finitezza e intervallo pixel.
   **Non abbiamo corretto il benchmark:** questo non dimostra che tutti i frame
   dei rollout siano immuni al caso grayscale; se incontrato, va analizzato e
   riportato, non nascosto con un reshape o saltando il task.

## Organizzazione dello scratch

```text
$SCRATCH_FLASH/TurboVLA/
├── simulators/LIBERO/                         invariato
├── simulators/LIBERO-plus/libero/libero/
│   ├── assets/                               nuovi asset estratti
│   ├── bddl_files/                           task ufficiali
│   ├── init_files/                           stati iniziali
│   └── benchmark/task_classification.json
├── simulator_assets/liberoplus/assets.zip     download ufficiale verificato
├── config/libero/config.yaml                 invariato
├── config/liberoplus/config.yaml             configurazione distinta
├── manifests/                               revisioni, estrazione, versioni env
├── models/                                  riusati, non duplicati
├── pretrained/                              riusati, non duplicati
├── training/libero/libero-baseline/           invariato
├── results/libero/                           confronti precedenti invariati
└── results/liberoplus/NOME_RUN/SUITE/         nuove evaluation
```

`datasets/liberoplus_unused` è solo un percorso non utilizzato richiesto dalla
config del simulatore; non viene creato/scaricato un dataset di training.
L'ambiente è in `/home/gpepe/miniconda3/envs/turbovla-liberoplus`, come gli altri.

## Stato

- **Aggiornamento 15 settembre 2026:** primo run GPU **1930382**,
  `liberoplus-author-ckpt-unified`, terminato per time limit l'11 settembre alle
  21:37:15 UTC. Spatial e Object sono complete e i conteggi dei JSON sono coerenti;
  Goal aveva completato 263 episodi nei log ma non ha un `results.json` finale.
  Long non era iniziata. Non si includono i tentativi parziali nel punteggio.

  | Suite | Successi / episodi | Success rate | Directory risultati |
  |---|---:|---:|---|
  | Spatial | 1.596 / 2.402 | 66,44% | `results/liberoplus/liberoplus-author-ckpt-unified/libero_spatial/` |
  | Object | 1.672 / 2.518 | 66,40% | `results/liberoplus/liberoplus-author-ckpt-unified/libero_object/` |
  | Goal | in attesa del nuovo run | — | `results/liberoplus/liberoplus-author-ckpt-unified-goal-long/libero_goal/` |
  | Long | in attesa del nuovo run | — | `results/liberoplus/liberoplus-author-ckpt-unified-goal-long/libero_10/` |

  Il job **1938718**, `liberoplus-author-ckpt-unified-goal-long`, è stato
  sottomesso per Goal dall'inizio, poi Long: una GPU A40, 8 CPU, 64 GiB, limite
  24 ore (massimo della partizione). Usa `sbatch --wrap` con due invocazioni
  sequenziali di `scripts/cluster/evaluate.sh --benchmark liberoplus --trials 1`,
  rispettivamente `--suite libero_goal` e `--suite libero_10`, interrotte se la
  prima fallisce. Nessun nuovo script operativo e nessuna ripresa dal task 263.
  Checkpoint unified, config della policy e codice Python di policy/evaluator
  identici al primo run; la durata resta da verificare per le due suite residue.
  Output derivato dal nome job, log in `logs/turbovla/evaluation/` con quel nome.
  Vecchi risultati, sorgenti parziali Goal e log del timeout conservati intatti.
- **57 test locali passati** (38 precedenti + 19 LIBERO+). Questi non sono rollout.
- Revisione locale conclusiva: controlli CPU ripetuti senza il checkout LIBERO+
  nel `PYTHONPATH`, manifest reale e 10.030 task ricontrollati, hash delle config
  TurboVLA rilasciate invariati; comandi generati verificati con il parser upstream.
  Nessuna nuova installazione o evaluation GPU lanciata per questa revisione.
- Job finale **1930349**, `liberoplus-env-retry-05`: **COMPLETED, exit 0:0**,
  `compute-7-4`, 10 settembre 2026, 21:19:52–21:21:44 UTC. `pip check`, librerie
  native, import LIBERO+, conteggi delle quattro suite e fog/motion blur RGB
  a 256 px alle severità 1/10 passati. Pip già coerente: il secondo avvio non
  lo ha reinstallato. Manifest salvati in `manifests/environments/`:
  `turbovla-liberoplus.conda-explicit.txt` e `turbovla-liberoplus.pip-freeze.txt`.
  Lo stderr contiene avvisi Gym e deprecazione `np.fromstring`, senza traceback.
  Al completamento di quel setup erano validati solo i controlli CPU; le prove
  GPU successive sono riportate nell'aggiornamento sopra.
- `liberoplus-assets-retry-03` ha completato l'estrazione e la verifica:
  **PASSED**, 448.799 file estratti, 10.030 task e quattro gruppi di asset.
  Configurazione e manifest di completamento sono presenti. Non serve riscaricare.
- Job ambiente **1930306**, `liberoplus-env-retry-03`: clone creato e librerie
  native verificate, poi interruzione su `ImportError: get_runnable_pip`.
  Diagnosi: pip 26.0.1 ripristinato da Conda sovrapposto ai file pip 26.2.1
  copiati dal sorgente. Le dipendenze aggiuntive e LIBERO+ non erano installati.
  Il launcher ora gestisce questa riparazione.
- Job **1930346**, `liberoplus-env-retry-04`: pip riparato, dipendenze e
  LIBERO+ installati, `pip check` passato; **FAILED, exit 1**, al controllo
  motion blur con la vecchia fixture grigia uniforme. Corretta esclusivamente
  la fixture del preflight come descritto sopra; nessuna modifica al simulatore.
- Il successivo job `liberoplus-assets-retry-02` ha scaricato l'archivio completo
  e clonato il simulatore alla revisione fissata. Il checksum ZIP coincide con
  quello riportato sopra. È poi fallito prima dell'estrazione con
  `ValueError: Unexpected ZIP root`: il nostro controllo richiedeva `assets/`
  direttamente alla radice, mentre lo ZIP contiene il prefisso di build ufficiale.
  **Estrattore corretto** per normalizzare quel prefisso senza modificare i file.
  Verificati in sola lettura tutti i 457.675 percorsi dello ZIP reale; estratti
  e confrontati cinque file campione in una directory temporanea poi rimossa.
  Questo controllo non sostituisce l'estrazione completa sullo scratch.
  L'estrazione completa, il manifest e la configurazione sono stati poi prodotti
  da `liberoplus-assets-retry-03`, come riportato sopra.
- I problemi BeeGFS dei primi tentativi sono riportati sotto come storico:
  il tentativo `retry-02` ha superato la creazione delle directory e il download.
- Job asset **1929807**, `liberoplus-assets`: **FAILED**, exit 1 dopo tre secondi
  su `compute-4-3`, prima di scaricare l'archivio. BeeGFS ha restituito
  `OSError: [Errno 121] Remote I/O error` creando `simulator_assets/liberoplus`.
- Job ambiente **1929835**, `liberoplus-env`: **CANCELLED / DependencyNeverSatisfied**,
  perché dipendeva da `afterok:1929807`. Non ha eseguito installazioni.
- L'errore è stato riprodotto anche da `login1`, fuori dal sandbox, sia creando
  `simulator_assets/liberoplus` sia creando `config/liberoplus`. Mount reale `rw`,
  circa 120 TiB disponibili e permessi del proprietario corretti: la causa
  precisa lato BeeGFS non è determinata. Non è risolvibile aggiornando Python.
- **`turbovla-liberoplus` è installato e validato a livello CPU.**
  L'ambiente originale conserva il simulatore LIBERO standard; il clone importa
  LIBERO+ con configurazione separata. Torch 2.3.1+cu121, Transformers 4.56.0,
  NumPy 1.26.4 e MuJoCo 2.3.7 sono invariati; nel clone sono presenti
  Wand 0.6.13 e scikit-image 0.24.0. Nessun file tracciato del simulatore è stato
  modificato. Il `pip freeze` dell'ambiente originale è invariato prima e dopo
  la riparazione del clone (SHA256
  `dd46c4e656a2d08153b37021fc56827f70db0be9adade27e09bc5cf6e369586b`).
- Nessun training LIBERO+ avviato; evaluation completa sulle quattro suite
  ancora da terminare. L'integrazione viene pubblicata sul branch
  `setup/cluster-reproduction` con messaggio di commit `LiberoPlus`.

### Secondo tentativo e diagnosi BeeGFS

Ritentato su richiesta il 10 settembre, escludendo il nodo del primo errore:

```bash
sbatch --job-name=liberoplus-assets-retry --exclude=compute-4-3 --time=01:00:00 \
  scripts/cluster/assets.sh --benchmark liberoplus
```

Job **1930020**, eseguito su **compute-7-7**, fallito con exit 1 in due secondi
(16:16:08–16:16:10 UTC). Stesso errore 121 creando la directory degli asset;
nessun archivio scaricato e nessun job ambiente rilanciato.
Log: [liberoplus-assets-retry.err](/home/gpepe/ws/logs/turbovla/assets/liberoplus-assets-retry.err).

Il kernel di `login1` conferma la risposta remota per i due percorsi previsti:

```text
2026-09-10 16:14:26 UTC
MkDirResp ownerID: 1 parentID: 8A-6A976405-1 name: liberoplus error code: Internal error
MkDirResp ownerID: 1 parentID: 60-6A976210-1 name: liberoplus error code: Internal error
```

La prima directory padre è `simulator_assets`, la seconda `config`, sotto
`/mnt/beegfs/gpepe/TurboVLA`. Entrambe appartengono all'utente gpepe e hanno
permesso di scrittura per il proprietario; le ACL non aggiungono restrizioni.
Il mount reale è `rw`. Il client espone entrambi i server di metadati e tutti
i 16 target di storage come `Online / Good`: questo attesta lo stato pubblicato,
non che ogni operazione di creazione riesca. Lo spazio totale indicato da `df`
non certifica spazio/inode/quota dei singoli server di metadati.

Hugging Face risponde correttamente alla richiesta della revisione fissata e
conferma dimensione/hash dell'archivio. L'errore viene riprodotto da un semplice
`mkdir`, indipendentemente da Python, Hugging Face, Conda e Slurm. Il punto di
fallimento è pertanto la **creazione remota della directory da parte di BeeGFS**.
Non è ancora identificata la causa interna al server: non possiamo affermare
che sia un disco pieno, una quota o una corruzione dei metadati. `quota -s -v`
non restituisce informazioni utili sulle quote BeeGFS; i tool amministrativi
`beegfs`/`beegfs-ctl` non sono installati nel PATH e il journal di sistema non
è leggibile dall'utente. Non sono stati modificati servizi o mount.

Da inoltrare agli amministratori: verificare il log del server di metadati
**node_meta_1 / ID 1**, alle ore sopra indicate, per le due parentID, inclusi
filesystem/inode del volume dei metadati, capacità di allocazione e stato delle
quote/pool. Il fatto che i dati occupino 617 TiB su 736 TiB non esclude un
problema nello spazio dei metadati, che è distinto:
[documentazione BeeGFS sui requisiti dello storage metadata](https://doc.beegfs.io/7.3.4/system_design/system_requirements.html).

Al momento di questi primi errori occorreva ripristinare le creazioni di
directory nello scratch o concordare un altro storage. Il successivo
`liberoplus-assets-retry-02` ha superato quel punto, incontrando invece il problema
di prefisso ZIP descritto nello Stato. Questi problemi sono stati poi superati
con i job documentati sopra: **i comandi di questa sezione sono storico della
diagnosi, non istruzioni da rilanciare adesso**. Conservare i log di errore per
gli amministratori.
Ambiente, GPU/rendering e caricamento policy vanno dichiarati pronti solo
dopo il completamento dei job e delle verifiche reali, non per il solo dry-run.

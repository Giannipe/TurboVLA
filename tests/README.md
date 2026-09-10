# Test locali di TurboVLA e dei launcher cluster

Questi test sono aggiunti nel nostro branch: proteggono gli adattamenti locali
e il loro allineamento alla ricetta pubblica. **Non servono per lanciare un
training o una evaluation e non vengono eseguiti automaticamente da `sbatch`.**
Sono utili prima di modificare launcher, caricamento dei pesi o ripresa EMA;
per questo li conserviamo.

## Cosa verificano

| File | Test | Copertura |
|---|---:|---|
| [test_cluster_cli.py](test_cluster_cli.py) | 24 | Selezione asset LIBERO/RoboTwin; default e override CLI; quattro suite sequenziali; dry-run senza download/sottomissioni; hash con file sintetici e rifiuto di percorsi non sicuri; nomi da Slurm senza ID/timestamp; protezioni output/resume; log su stdout/stderr senza duplicati; copie dei sorgenti; launcher Bash e layout spool Slurm simulato; risorse/nomi job posizionati prima dello script; controllo single-node/single-task e GPU evaluation; configurazione RoboTwin e streaming dei suoi log. |
| [test_libero_training_recipe.py](test_libero_training_recipe.py) | 3 | Confronta i flag generati con il parser reale del training pubblico, adattando i percorsi locali e l'orizzonte LR equivalente. Verifica batch globale 128 su quattro GPU; variante batch 256 che cambia solo il batch per GPU; configurazione breve che mantiene schedule 80k, worker e accumulo. |
| [test_libero_export_compat.py](test_libero_export_compat.py) | 7 | Priorità ai pesi EMA; rifiuto di contenitori non validi o raw non verificati; accettazione delle eccezioni SHA256 per export unified/legacy senza trasformare i pesi; nome file e metadati da soli non bastano. Usa fixture sintetiche e hash autorizzati simulati. |
| [test_training_ema_resume.py](test_training_ema_resume.py) | 4 | Formula EMA con decay 0,999; continuità esatta di AdamW/EMA in un piccolo esempio CPU interrotto e ripreso; rifiuto di EMA mancante o forma incompatibile. |

Totale: **38 test**. Alcuni contengono più sottocasi. Il test RoboTwin usa
processi/simulatori finti: non è una evaluation robotica. Il test della ricetta
breve controlla solo gli argomenti, non avvia tre step di DDP.

## Come eseguirli

Dalla root della repository, con l'ambiente LIBERO già installato:

```bash
cd /home/gpepe/ws/TurboVLA
/home/gpepe/miniconda3/envs/turbovla-libero/bin/python -B -m unittest discover -s tests -v
```

In alternativa, `conda activate turbovla-libero` e lo stesso comando con `python`.
Si usa `unittest`, già incluso in Python; non serve installare pytest.
L'ambiente LIBERO serve perché alcuni moduli importano Torch, Transformers e
altre dipendenze del progetto. È previsto anche l'eseguibile `torchrun` nello
stesso ambiente. Non usare il Python di sistema senza queste dipendenze.

Per un solo gruppo, ad esempio EMA:

```bash
python -B -m unittest discover -s tests -p 'test_training_ema_resume.py' -v
```

Esito corretto: `Ran 38 tests` e `OK` per la suite completa, codice di uscita 0.
Warning di libreria in importazione non equivalgono a un test fallito:
controllare eventuali `FAIL`/`ERROR` e il riepilogo conclusivo.

Non occorrono GPU, rete o un'allocazione Slurm: non vengono chiamati `sbatch`,
training reali, rollout o installazioni. I file di prova sono temporanei e
vengono rimossi alla fine; non si modificano dataset/checkpoint nello scratch.
Gli import possono inizializzare cache di libreria. `-B` evita i `.pyc`.

## Cosa NON dimostrano

- Non verificano che Slurm avvii il job, che i mount siano accessibili sui nodi,
  che CUDA/DDP funzionino o che memoria e wall-time siano sufficienti.
- I test sintetici degli hash non certificano i pesi/dataset realmente scaricati.
  Per questi usare `sbatch scripts/cluster/assets.sh --benchmark all --verify-only --online`
  e controllarne l'esito; `--online` confronta gli indici remoti delle revisioni fissate.
- Non misurano success rate, equivalenza al paper o assenza di instabilità MuJoCo.
- Il piccolo test EMA non prova una ripresa distribuita bit-per-bit, né il
  ripristino completo dell'ordine dati/RNG del training.
- Non validano il simulatore RoboTwin o la sua installazione separata.

Le prove reali completate e i warning sono nel
[confronto LIBERO](../scripts/cluster/LIBERO_RESULTS.md); l'organizzazione dei
launcher e dello scratch è nel [README cluster](../scripts/cluster/README.md).

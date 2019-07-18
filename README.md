# ParallelPyTorchTabular

Exploring parallel deep learning for tabular data.

## Options

Type `python main.py --help` for options:

```
optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        input batch size for training (default: 80960)
  --epochs EPOCHS       number of epochs to train (default: 1)
  --batched             Use batched dataloader.
  --par PAR             Data-parallelism framework to use (`hvd`, `bps`,
                        `hog`).
  --hogwild-procs HOGWILD_PROCS
                        Use hogwild with this many processes.
  --hogwild-gpus HOGWILD_GPUS
                        Use distributed hogwild with this many gpus.
  --cpu-params          Keep shared-model on CPU for hogwild.
  --adam                Use adam optimizer instead of SGD.
  --base-lr BASE_LR, --lr BASE_LR
                        learning rate for a single GPU
  --warmup-epochs WARMUP_EPOCHS
                        number of warmup epochs
  --momentum MOMENTUM   SGD momentum
  --wd WD               weight decay
  --no-cuda             disables CUDA training
  --seed SEED           random seed
  --fp16-allreduce      use fp16 compression during allreduce ops
  --batches-per-allreduce BATCHES_PER_ALLREDUCE
                        In hrovod or byteps, number of batches processed
                        locally before executing allreduce across workers; it
                        multiplies total batch size.
  --data-dir DATA_DIR   Dataset location
```


## Use Examples

### Hogwild: Multiprocessed-Asynchronous Learning (Single GPU)

Use `--par hog --hogwild-procs <number of desired processes>`:

**E.g.**
```
$ python main.py --epochs 3 --par hog --hogwild-procs 4 --adam
```

### Hogwild: Multiprocessed-Asynchronous Learning (Multiple GPUs)

Use `--par hog --hogwild-procs <number of desired processes>` and `--hogwild-gpus <gpu count>` to distribute hogwild-processes accross multiple GPUs:

**E.g.**
```
$ python main.py --epochs 3 --par hog --hogwild-procs 4 --hogwild-gpus 4 --adam
```

### Horovod: Distributed Synchronous Learning (Multiple GPUs)

Use `mpirun -n <number of desired processes>` and `--par hvd` to perform Horovod-distributed learning:

**E.g.**
```
$ mpirun -n 4 python main.py --epochs 3 --par hvd --adam
```

### BytePS: Distributed Parameter-Server Learning (Multiple GPUs)

Use `mpirun -n <number of desired processes>` and `--par bps` to perform BytePS-distributed learning:

**E.g.**
```
python launch_bps.py --par bps --adam --epochs 3
```

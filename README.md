# ParallelPyTorchTabular

Exploring parallel deep learning for tabular data

## Hogwild: Multiprocessed-Asynchronous Learning (Single GPU)

Use `--hogwild <number of desired processes>`:

**e.g.**
```
python parallel_main.py --epochs 3 --hogwild 4 --hogwild-gpus 4
```

## Hogwild: Multiprocessed-Asynchronous Learning (Multiple GPUs)

Use `--hogwild <number of desired processes>` and `--hogwild-gpus <gpu count>` to distribute hogwild-processes accross multiple GPUs:

**e.g.**
```
python parallel_main.py --epochs 3 --hogwild 4 --hogwild-gpus 4
```

## Horovod: Distributed Synchronous Learning (Multiple GPUs)

Use `mpirun -n <number of desired processes>` and `--hvd` to perform horovod-distributed learning:

**e.g.**
```
mpirun -n 4 python parallel_main.py --epochs 3 --hvd
```


import argparse
import time
import os

# ========================================================================== #
#                                                                            #
#  Input/Options                                                             #
#                                                                            #
# ========================================================================== #

parser = argparse.ArgumentParser(description="Parallel Mortgage Workflow")
parser.add_argument("--batch-size", type=int, default=80960, help="input batch size for training (default: 80960)")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 1)")
parser.add_argument("--batched", action="store_true", default=False, help="Use batched dataloader.")
parser.add_argument("--hvd", action="store_true", default=False, help="Use horovod for data parallelism.")
parser.add_argument("--hogwild", type=int, default=0, help="Use hogwild with this many processes.")
parser.add_argument("--hogwild-gpus", type=int, default=1, help="Use distributed hogwild with this many gpus.")
parser.add_argument("--cpu-params", action="store_true", default=False, help="Keep shared-model on CPU for hogwild.")
parser.add_argument("--adam", action="store_true", default=False, help="Use adam optimizer instead of SGD.")
parser.add_argument("--base-lr", "--lr", type=float, default=0.01, help="learning rate for a single GPU")
parser.add_argument("--warmup-epochs", type=float, default=5, help="number of warmup epochs")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=0.00005, help="weight decay")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--fp16-allreduce", action="store_true", default=False, help="use fp16 during hvd allreduce")
parser.add_argument("--batches-per-allreduce", type=int, default=1,
    help="In hrovod, number of batches processed locally "
    "before executing allreduce across workers; it multiplies "
    "total batch size.",
)
parser.add_argument("--data-dir", default="/datasets/mortgage/post_etl/dnn/", help="Dataset location")
# Note: In Docker, use "--data-dir /data/mortgage/"
args = parser.parse_args()

# Hard-coded options
args.num_features = 2 ** 22 # When hashing features range will be [0, args.num_features)
args.embedding_size = 64
args.hidden_dims = [600, 600, 600, 600]
args.num_samples = 8096000
args.log_interval = 100

# Assume 8-GPUs if CUDA_VISIBLE_DEVICES not set
cuda_visible = os.environ.get("cuda_visible", "0,1,2,3,4,5,6,7")
if isinstance(cuda_visible, str):
    cuda_visible = cuda_visible.split(",")
cuda_visible = list(map(int, cuda_visible))
args.gpu_ids = cuda_visible[:args.hogwild_gpus]


# ========================================================================== #
#  Main "Method"                                                             #
# ========================================================================== #
if __name__ == '__main__':

    if args.hogwild > 0:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
        # Hogwild gets preference over horovod, for now.
        # Can try to combine methodologies in the future (or detect which is better)
        if args.hvd:
            args.hvd = False
            print("WARNING - Not using Horovod (Using Hogwild).")
        if not args.batched:
            args.batched = True
            print("WARNING - Switching to batched data loader.")
        if args.batches_per_allreduce > 1:
            args.batches_per_allreduce = 1
            print("WARNING - Using batches_per_allreduce = 1.")
        if not args.adam:
            args.adam = True
            print("WARNING - SharedAdam currently required for hogwild.")
    import torch
    from torch import nn
    import torch.optim as torch_optim
    if args.hvd:
        import horovod.torch as hvd

    from model import MortgageNetwork
    from parallel_training import train
    from shared_optimizer import SharedAdam


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.gpu_ids = [-1]

    # Initialize Horovod/Cuda
    myrank = 0
    mysize = 1
    if args.hvd:
        hvd.init()
        myrank = hvd.rank()
        mysize = hvd.size()
    torch.manual_seed(args.seed)
    if args.cuda:
        # Horovod: pin GPU to local rank.
        if args.hvd:
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(args.seed)

    # Model definition
    model = MortgageNetwork(
        args.num_features,
        args.embedding_size,
        args.hidden_dims,
        activation=nn.ReLU(),
        use_cuda=args.cuda and not args.cpu_params,
    )

    if args.hogwild > 0:
        model.share_memory() # gradients are allocated lazily, so they are not shared here

    # Using Adam optimizer?
    if args.adam:
        if args.hogwild > 0 and args.hogwild_gpus > 1:
            # Use distributed optimizer for distributed hogwild
            optimizer = SharedAdam(model.parameters(), lr=args.base_lr)
            optimizer.share_memory()
        else:
            optimizer = torch_optim.Adam(model.parameters(), lr=args.base_lr)
    else:
        # Horovod: scale learning rate by the number of GPUs.
        # Gradient Accumulation: scale learning rate by batches_per_allreduce
        optimizer = torch_optim.SGD(
            model.parameters(),
            lr=(args.base_lr * args.batches_per_allreduce * mysize),
            momentum=args.momentum,
            weight_decay=args.wd,
        )

    processes = []
    start_time = time.time()
    if args.hogwild > 0:
        for rank in range(args.hogwild):
            myrank = rank
            mysize = args.hogwild
            p = mp.Process(target=train, args=(myrank, mysize, model, optimizer, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(myrank, mysize, model, optimizer, args)
    end_time = time.time()

    if myrank==0:
        print("Total Training Time: "+str(end_time - start_time)+" seconds")

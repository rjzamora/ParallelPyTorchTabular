import argparse
import time
import sys
import os

import warnings
warnings.filterwarnings('ignore')

# ========================================================================== #
#                                                                            #
#  Input/Options                                                             #
#                                                                            #
# ========================================================================== #

parser = argparse.ArgumentParser(description="Parallel Mortgage Workflow")
parser.add_argument("--batch-size", type=int, default=80960, help="input batch size for training (default: 80960)")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 1)")
parser.add_argument("--batched", action="store_true", default=False, help="Use batched dataloader.")
parser.add_argument("--par", default=None, help="Data-parallelism framework to use (`hvd`, `bps`, `hog`).")
parser.add_argument("--hogwild-procs", type=int, default=1, help="Use hogwild with this many processes.")
parser.add_argument("--hogwild-gpus", type=int, default=1, help="Use distributed hogwild with this many gpus.")
parser.add_argument("--gpu-params", action="store_true", default=False, help="Keep shared-model on GPU for hogwild.")
parser.add_argument("--adam", action="store_true", default=False, help="Use adam optimizer instead of SGD.")
parser.add_argument("--base-lr", "--lr", type=float, default=0.01, help="learning rate for a single GPU")
parser.add_argument("--warmup-epochs", type=float, default=5, help="number of warmup epochs")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=0.00005, help="weight decay")
parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--fp16-allreduce", action="store_true", default=False, help="use fp16 compression during allreduce ops")
parser.add_argument("--batches-per-allreduce", type=int, default=1,
    help="In hrovod or byteps, number of batches processed locally "
    "before executing allreduce across workers; it multiplies "
    "total batch size.",
)
parser.add_argument("--data-dir", default="/datasets/mortgage/post_etl/dnn/", help="Dataset location")
# Note: In Docker, use "--data-dir /data/mortgage/"
args = parser.parse_args()

# Hard-coded options
args.num_features = (
    2 ** 22
)  # When hashing features range will be [0, args.num_features)
args.embedding_size = 64
args.hidden_dims = [600, 600, 600, 600]
args.num_samples = 8662720  # Used to limit data in conventional dataloader
args.num_files = 1 # Used to limit data in batched dataloader
args.log_interval = 1000

# Assume 8-GPUs if CUDA_VISIBLE_DEVICES not set
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
if isinstance(cuda_visible, str):
    cuda_visible = cuda_visible.split(",")
cuda_visible = list(map(int, cuda_visible))
args.hogwild_gpus = min(args.hogwild_gpus, len(cuda_visible))
args.gpu_ids = [i for i in range(args.hogwild_gpus)]


# ========================================================================== #
#  Main "Method"                                                             #
# ========================================================================== #
if __name__ == "__main__":

    # Check if we need to import multiprocessing
    if args.par and args.par == "hog":
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    import torch
    from torch import nn
    import torch.optim as torch_optim

    from model import MortgageNetwork
    from training import train
    from shared_optimizer import SharedAdam

    # Check if we are doing data parallelism
    if args.par:
        if args.par == "hvd":
            # Using Horovod to synchronize gradients accross multiple workers
            import horovod.torch as hvd

        elif args.par == "bps":
            # Using BytePS to synchronize gradients accross multiple workers
            import byteps.torch as bps

        elif args.par == "hog":
            # Using Hogwild for asynchronous multiprocessing
            if not args.batched:
                args.batched = True
                print(
                    "WARNING - Switching to batched data loader.\n"
                    "Hogwild cannot currently handle multi-worker dataloader."
                )
            if args.batches_per_allreduce > 1:
                args.batches_per_allreduce = 1
                print("WARNING - Using batches_per_allreduce = 1.")
            if not args.adam and args.hogwild_gpus > 1:
                args.adam = True
                print("WARNING - SharedAdam required for distributed hogwild.")

        else:
            raise ValueError("--par input must be in [`hvd`, `bps`, `hog`]")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.gpu_ids = [-1]
    elif args.par == "hog":
        if not args.gpu_params and args.hogwild_gpus <= 1:
            args.gpu_params = True
            # Note: Single-GPU hogwild needs to keep the model on the GPU (for now)

    # Set model device
    if args.par == "hog":
        if args.cuda and args.gpu_params:
            args.model_device = torch.device(type='cuda', index=args.gpu_ids[0])
        else:
            args.model_device = torch.device('cpu')
    else:
        if args.cuda:
            args.model_device = torch.device('cuda')
        else:
            args.model_device = torch.device('cpu')

    # Initialize Horovod/Cuda
    myrank = 0
    mysize = 1
    if args.par == "hvd":
        hvd.init()
        myrank = hvd.rank()
        mysize = hvd.size()
    elif args.par == "bps":
        bps.init()
        myrank = bps.rank()
        mysize = bps.size()
    torch.manual_seed(args.seed)
    if args.cuda:
        # Horovod & BytePS: pin GPU to local rank.
        if args.par == "hvd":
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(args.seed)
        if args.par == "bps":
            torch.cuda.set_device(bps.local_rank())
            torch.cuda.manual_seed(args.seed)

    # Model definition
    model = MortgageNetwork(
        args.num_features,
        args.embedding_size,
        args.hidden_dims,
        activation=nn.ReLU(),
        device=args.model_device,
    )

    # Using Adam optimizer?
    if args.adam:
        if args.par == "hog" and args.hogwild_gpus > 1:
            # Use distributed optimizer for distributed hogwild
            optimizer = SharedAdam(model.parameters(), lr=args.base_lr)
            optimizer.share_memory()
        else:
            optimizer = torch_optim.Adam(model.parameters(), lr=args.base_lr)
    else:
        # Horovod and BytePS: Scale learning rate by the number of GPUs.
        # Gradient Accumulation: scale learning rate by batches_per_allreduce
        optimizer = torch_optim.SGD(
            model.parameters(),
            lr=(args.base_lr * args.batches_per_allreduce * mysize),
            momentum=args.momentum,
            weight_decay=args.wd,
        )

    processes = []
    start_time = time.time()
    if args.par == "hog":
        for rank in range(args.hogwild_procs):
            myrank = rank
            mysize = args.hogwild_procs
            p = mp.Process(target=train, args=(myrank, mysize, model, optimizer, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(myrank, mysize, model, optimizer, args)
    end_time = time.time()

    if myrank == 0:
        print("\n\tTotal Training Time: " + str(end_time - start_time) + " seconds\n")
    sys.exit(0)

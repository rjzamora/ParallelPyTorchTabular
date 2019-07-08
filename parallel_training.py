import argparse
import datetime as dt
import glob, os, re, subprocess, tempfile
from tqdm import tqdm
import time
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils import data as torch_data
import torch.multiprocessing as mp

from dataset_from_parquet import dataset_from_parquet
from batch_dataset_from_parquet import batch_dataset_from_parquet
import batch_dataset, batch_dataloader
from model import MortgageNetwork

import horovod.torch as hvd


# ========================================================================== #
#                                                                            #
#  Input/Options                                                             #
#                                                                            #
# ========================================================================== #

parser = argparse.ArgumentParser(description="Parallel Mortgage Workflow")
parser.add_argument("--batch-size", type=int, default=80960, help="input batch size for training (default: 80960)")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train (default: 1)")
parser.add_argument("--base-lr", type=float, default=0.01, help="learning rate for a single GPU")
parser.add_argument("--warmup-epochs", type=float, default=5, help="number of warmup epochs")
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=0.00005, help="weight decay")
parser.add_argument("--batched", action="store_true", default=False, help="Use batched dataloader.")
parser.add_argument("--hvd", action="store_true", default=False, help="Use horovod for data parallelism.")
parser.add_argument("--hogwild", type=int, default=0, help="Use hogwild with this many processes.")
parser.add_argument("--adam", action="store_true", default=False, help="Use adam optimizer instead of SGD.")
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
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

# Hogwild gets preference over horovod, for now.
# Can try to combine methodologies in the future (or detect which is better)
if args.hvd and args.hogwild > 0:
    args.hvd = False

# Hard-coded options
num_features = 2 ** 22  # When hashing features range will be [0, num_features)
embedding_size = 64
num_samples = 8096000
hidden_dims = [600, 600, 600, 600]


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, optimizer, loader_size):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / loader_size
        lr_adj = (
            1.0
            / hvd.size()
            * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
        )
    elif epoch < 30:
        lr_adj = 1.0
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group["lr"] = (
            args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj
        )


# Average metrics from distributed (horovod) training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val):
        if args.hvd:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        else:
            self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


# ========================================================================== #
#                                                                            #
#  Training Function Definitions                                             #
#                                                                            #
# ========================================================================== #

def train_epoch(model, loss_fn, optimizer, train_loader, train_sampler, epoch, verbose):
    model.train()
    if not args.batched:
        train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss")
    num_iterations = int(len(train_loader) // max(1, mysize))
    with tqdm(
        total=num_iterations,
        desc="Train Epoch     #{}".format(epoch + 1),
        disable=not verbose,
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == num_iterations:
                break
            if not args.adam:
                adjust_learning_rate(epoch, batch_idx, optimizer, len(train_loader))
            if args.cuda:
                data = data.to("cuda")
                target = target.to("cuda")
            optimizer.zero_grad()

            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i : i + args.batch_size]
                target_batch = target[i : i + args.batch_size]
                output = model(data_batch)
                loss = loss_fn(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({"loss": train_loss.avg.item()})
            t.update(1)


def train(model, myrank, mysize):

    # ====================================================================== #
    #  Data Handling                                                         #
    # ====================================================================== #

    out_dir = args.data_dir
    kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}
    if args.batched:
        train_dataset = batch_dataset_from_parquet(
            os.path.join(out_dir, "train"),
            num_files=1,
            batch_size=allreduce_batch_size,
            use_cuDF=False,
            use_GPU_RAM=False,
        )
        train_sampler = None
        train_loader = batch_dataloader.BatchDataLoader(
            train_dataset, shuffle=True, mysize=mysize, myrank=myrank
        )
    else:
        train_dataset = dataset_from_parquet(
            os.path.join(out_dir, "train"),
            num_samples=num_samples,
            shuffle_files=False,
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=mysize, rank=myrank
        )
        train_loader = torch_data.DataLoader(
            train_dataset,
            batch_size=allreduce_batch_size,
            sampler=train_sampler,
            **kwargs
        )

    # ====================================================================== #
    #  Define Loss/Optimization                                              #
    # ====================================================================== #

    # Using Adam optimizer?
    if args.adam:
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

    if args.hvd:
        # Horovod: (optional) compression algorithm.
        compression = (
            hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
        )

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=args.batches_per_allreduce,
        )

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Loss Function
    loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)

    # ====================================================================== #
    #  Main Training Loop                                                    #
    # ====================================================================== #

    verbose = 1 if myrank == 0 else 0
    start_time = time.time()
    for epoch in range(args.epochs):
        train_epoch(model, loss_fn, optimizer, train_loader, train_sampler, epoch, verbose)

    # Print final performance summary
    # For hogwild, other ranks may still be working...
    if verbose:
        if args.batched:
            num_iterations = len(train_loader)
        else:
            num_iterations = len(train_loader) * mysize
        ex_per_epoch = num_iterations * allreduce_batch_size
        total_examples = args.epochs * ex_per_epoch
        total_time = time.time() - start_time
        print(
            "\n{} Epochs in {} seconds ({} examples per epoch) -> [{} examples/s]".format(
                args.epochs, total_time, ex_per_epoch, total_examples / total_time
            )
        )


# ========================================================================== #
#  Main "Method"                                                             #
# ========================================================================== #
if __name__ == '__main__':

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
        num_features,
        embedding_size,
        hidden_dims,
        activation=nn.ReLU(),
        use_cuda=args.cuda,
    )

    if args.hogwild > 0:
        if args.cuda:
            model = model.to("cuda")
        model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    start_time = time.time()
    if args.hogwild > 0:
        for rank in range(args.hogwild):
            myrank = rank
            mysize = args.hogwild
            p = mp.Process(target=train, args=(myrank, mysize, model))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(myrank, mysize, model)
    end_time = time.time()

    if myrank==0:
        print("Total Training Time: "+str(end_time - start_time)+" seconds")

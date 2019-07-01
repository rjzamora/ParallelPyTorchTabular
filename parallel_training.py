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

from dataset_from_parquet import dataset_from_parquet
from batch_dataset_from_parquet import batch_dataset_from_parquet
import batch_dataset, batch_dataloader
from model import MortgageNetwork

import tensorboardX
import horovod.torch as hvd


# ========================================================================== #
#                                                                            #
#  Input/Options                                                             #
#                                                                            #
# ========================================================================== #

parser = argparse.ArgumentParser(description="Parallel Mortgage Workflow")
parser.add_argument(
    "--batch-size",
    type=int,
    default=80960,
    help="input batch size for training (default: 80960)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 1)",
)
parser.add_argument(
    "--base-lr",
    type=float,
    default=0.01,
    help="learning rate for a single GPU",
)
parser.add_argument(
    "--warmup-epochs", type=float, default=5, help="number of warmup epochs"
)
parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
parser.add_argument("--wd", type=float, default=0.00005, help="weight decay")
parser.add_argument(
    "--batched",
    action="store_true",
    default=False,
    help="Use batched dataloader.",
)
parser.add_argument(
    "--hvd",
    action="store_true",
    default=False,
    help="Use horovod for data parallelism.",
)
parser.add_argument(
    "--checkpoint",
    action="store_true",
    default=False,
    help="enable checkpointing and restarting.",
)
parser.add_argument(
    "--adam",
    action="store_true",
    default=False,
    help="Use adam optimizer instead of SGD.",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training",
)
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)
parser.add_argument(
    "--checkpoint-format",
    default="./checkpoint-{epoch}.pth.tar",
    help="checkpoint file format",
)
parser.add_argument(
    "--log-dir", default="./logs", help="tensorboard log directory"
)
parser.add_argument(
    "--batches-per-allreduce",
    type=int,
    default=1,
    help="number of batches processed locally before "
    "executing allreduce across workers; it multiplies "
    "total batch size.",
)
parser.add_argument(
    "--data-dir",
    default="/datasets/mortgage/post_etl/dnn/",
    help="Dataset location",
)  # Docker: "/data/mortgage/"
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

# Hard-coded options
num_features = 2 ** 22  # When hashing features range will be [0, num_features)
embedding_size = 64
num_samples = 8096000
hidden_dims = [600, 600, 600, 600]

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

# Start from checkpoint (optional)
resume_from_epoch = 0
if args.checkpoint:
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
    if args.hvd:
        # Horovod: broadcast resume_from_epoch from rank 0 (which will have
        # checkpoints) to other ranks.
        resume_from_epoch = hvd.broadcast(
            torch.tensor(resume_from_epoch),
            root_rank=0,
            name="resume_from_epoch",
        ).item()

# Printing and logging
verbose = 1 if myrank == 0 else 0
log_writer = tensorboardX.SummaryWriter(args.log_dir) if myrank == 0 else None

# ========================================================================== #
#                                                                            #
#  Data Handling                                                             #
#                                                                            #
# ========================================================================== #

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

# ========================================================================== #
#                                                                            #
#  Define Model and Optimization                                             #
#                                                                            #
# ========================================================================== #

# Define Model
model = MortgageNetwork(
    num_features,
    embedding_size,
    hidden_dims,
    activation=nn.ReLU(),
    use_cuda=args.cuda,
)

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

# Horovod: (optional) compression algorithm.
compression = (
    hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
)

if args.hvd:
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
    )

if args.checkpoint:
    # Restore from a previous checkpoint, if initial_epoch is specified.
    if resume_from_epoch > 0 and myrank == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

if args.hvd:
    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Loss Function
loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)

# ========================================================================== #
#                                                                            #
#  Model Training                                                            #
#                                                                            #
# ========================================================================== #

# Define function to train the model
def train(epoch):
    #model.train()
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
                adjust_learning_rate(epoch, batch_idx)
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

    if log_writer:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)


# TODO: Add validate() function


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
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


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
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


# MAIN Training Loop
start_time = time.time()
for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    if args.checkpoint:
        save_checkpoint(epoch)


# Print final performance summary
if myrank == 0:
    if args.batched:
        num_iterations = len(train_loader)
    else:
        num_iterations = len(train_loader) * mysize
    ex_per_epoch = num_iterations * allreduce_batch_size
    total_epochs = args.epochs - resume_from_epoch
    total_examples = total_epochs * ex_per_epoch
    total_time = time.time() - start_time
    print(
        "\n{} Epochs in {} seconds ({} examples per epoch) -> [{} examples/s]".format(
            total_epochs, total_time, ex_per_epoch, total_examples / total_time
        )
    )

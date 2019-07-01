import argparse
import datetime as dt
import glob, os, re, subprocess, tempfile
import time

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils import data as torch_data

from dataset_from_parquet import dataset_from_parquet
from batch_dataset_from_parquet import batch_dataset_from_parquet
import batch_dataset, batch_dataloader
from model import MortgageNetwork

import horovod.torch as hvd

# ========================================================================== #
#                                                                            #
#  Parse Input                                                               #
#                                                                            #
# ========================================================================== #

parser = argparse.ArgumentParser(description='Parallel Mortgage Workflow')
parser.add_argument('--batch-size', type=int, default=80960, metavar='N',
                    help='input batch size for training (default: 80960)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

num_features = 2 ** 22  # When hashing features range will be [0, num_features)
embedding_size = 64
hidden_dims = [600,600,600,600]
activation = nn.ReLU()
using_docker = False
device = "cuda"
num_workers=8
batch_dataload=False
use_cuDF=False
use_GPU_RAM=False

# Define data location
if using_docker:
    data_dir = "/data/mortgage/"
else:
    data_dir = "/datasets/mortgage/post_etl/dnn/"

# Initialize Horovod
hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

# ========================================================================== #
#                                                                            #
#  Define Model                                                              #
#                                                                            #
# ========================================================================== #

# Define Model
model = MortgageNetwork(
    num_features,
    embedding_size,
    hidden_dims,
    activation=activation,
    use_cuda=args.cuda
)

# ========================================================================== #
#                                                                            #
#  Define Data-loaders                                                       #
#                                                                            #
# ========================================================================== #

# Data
train_batch_size = args.batch_size
validation_batch_size = train_batch_size * 2
log_interval = 250 * 2048 // train_batch_size
out_dir = data_dir

if batch_dataload:
    train_dataset = batch_dataset_from_parquet(
        os.path.join(out_dir, "train"),
        num_files=1,
        batch_size=train_batch_size,
        use_cuDF=use_cuDF,
        use_GPU_RAM=use_GPU_RAM,
    )
    # validation_dataset = batch_dataset_from_parquet(
    #     os.path.join(out_dir, "validation"),
    #     batch_size=validation_batch_size,
    #     use_cuDF=use_cuDF,
    #     use_GPU_RAM=False,
    #     num_files=3,
    # )
    # test_dataset = batch_dataset_from_parquet(
    #     os.path.join(out_dir, "test"),
    #     batch_size=validation_batch_size,
    #     use_cuDF=use_cuDF,
    #     use_GPU_RAM=False,
    #     num_files=3,
    # )
    train_loader = batch_dataloader.BatchDataLoader(
        train_dataset, shuffle=True
    )
    # validation_loader = batch_dataloader.BatchDataLoader(
    #     validation_dataset, shuffle=False
    # )
    # test_loader = batch_dataloader.BatchDataLoader(
    #     test_dataset, shuffle=False
    # )
else:
    train_dataset = dataset_from_parquet(
        os.path.join(out_dir, "train"), num_samples=None, shuffle_files=False
    )
    # validation_dataset = dataset_from_parquet(
    #     os.path.join(out_dir, "validation")
    # )
    # test_dataset = dataset_from_parquet(os.path.join(out_dir, "test"))

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler
    )
    # validation_loader = torch_data.DataLoader(
    #     validation_dataset,
    #     batch_size=validation_batch_size,
    #     num_workers=num_workers,
    # )
    # test_loader = torch_data.DataLoader(
    #     test_dataset,
    #     batch_size=validation_batch_size,
    #     num_workers=num_workers,
    # )

# ========================================================================== #
#                                                                            #
#  Optimization Options                                                      #
#                                                                            #
# ========================================================================== #

# Optimizer
#optimizer = torch_optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch_optim.SGD(model.parameters(), lr=args.lr * hvd.size(),
                            momentum=args.momentum)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)

# Loss Function
loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(
    pred, target
)

# ========================================================================== #
#                                                                            #
#  Model Training                                                            #
#                                                                            #
# ========================================================================== #

# Train the model
total_step = len(train_loader)
for epoch in range(args.epochs):
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                .format(epoch+1, args.epochs, i+1, total_step, loss.item()))

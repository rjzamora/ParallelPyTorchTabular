import os
import sys
import time
import math
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as torch_optim
from torch.utils import data as torch_data

from dataset_from_parquet import dataset_from_parquet
from batch_dataset_from_parquet import batch_dataset_from_parquet
import batch_dataset, batch_dataloader

from model import MortgageNetwork


try:
    import horovod.torch as hvd
except:
    hvd = False

try:
    import byteps.torch as bps
except:
    bps = False


def ensure_shared_grads(model, shared_model, model_device):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        try:
            shared_param._grad = param.grad.to(device=model_device)
        except:
            print("Failed to copy local to shared gradients.")
            raise (RuntimeError)


# Horovod: using `lr = base_lr * mysize` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * mysize` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, optimizer, loader_size, args):
    if args.par == "hvd":
        mysize = hvd.size()
    elif args.par == "bps":
        mysize = bps.size()
    else:
        mysize = 1
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / loader_size
        lr_adj = 1.0 / mysize * (epoch * (mysize - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.0
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.base_lr * mysize * args.batches_per_allreduce * lr_adj


# ========================================================================== #
#                                                                            #
#  Training Function Definition                                              #
#                                                                            #
# ========================================================================== #


def train_epoch(
    model, local_model, loss_fn, optimizer, train_loader, train_sampler, epoch, args
):
    if not args.par == "hog":
        model.train()
    if not args.batched:
        train_sampler.set_epoch(epoch)
    num_iterations = int(len(train_loader) // max(1, args.mysize))
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx == num_iterations:
            break
        if args.par in ["hvd", "bps"] and not args.adam:
            adjust_learning_rate(
                epoch, batch_idx, optimizer, len(train_loader), args
            )

        if args.par == "hog":
            if args.hogwild_gpus > 1:
                local_model.zero_grad()
                local_model.load_state_dict(model.state_dict())
                output = local_model(data.cuda())
                loss = loss_fn(output, target.cuda())
                loss.backward()
                ensure_shared_grads(local_model, model, args.model_device)
            else:
                if args.cuda:
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
        else:
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i : i + args.batch_size]
                target_batch = target[i : i + args.batch_size]
                output = model(data_batch)
                loss = loss_fn(output, target_batch)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

        # Gradient is applied across all ranks
        optimizer.step()

        if args.verbose and (batch_idx % args.log_interval == 0 or batch_idx == num_iterations - 1):
            print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1), num_iterations,
                100. * (batch_idx+1) / num_iterations, loss.item()))
            sys.stdout.flush()


def train(myrank, mysize, model, optimizer, args):

    # ====================================================================== #
    #  Data Handling                                                         #
    # ====================================================================== #

    distributed_hw = False
    if args.par == "hog":
        if args.hogwild_gpus > 1:
            distributed_hw = True
            gpu_id = args.gpu_ids[myrank % len(args.gpu_ids)]
        else:
            gpu_id = args.gpu_ids[0]
        torch.manual_seed(args.seed + myrank)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed + myrank)
    else:
        gpu_id = -1

    out_dir = args.data_dir
    allreduce_batch_size = args.batch_size * args.batches_per_allreduce
    kwargs = {"num_workers": 8, "pin_memory": True} if args.cuda else {}
    if args.batched:
        train_dataset = batch_dataset_from_parquet(
            os.path.join(out_dir, "train"),
            num_files=args.num_files,
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
            num_samples=args.num_samples,
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
    #  Define Loss/Optimization/LocalModel                                   #
    # ====================================================================== #

    if args.par == "hvd":
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

    elif args.par == "bps":
        # BytePS: (optional) compression algorithm.
        compression = (
            bps.Compression.fp16 if args.fp16_allreduce else bps.Compression.none
        )

        # BytePS: wrap optimizer with DistributedOptimizer.
        optimizer = bps.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=args.batches_per_allreduce,
        )

        # BytePS: broadcast parameters & optimizer state.
        bps.broadcast_parameters(model.state_dict(), root_rank=0)
        bps.broadcast_optimizer_state(optimizer, root_rank=0)

    # Loss Function
    loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target)

    local_model = None
    if distributed_hw:
        # Local model definition
        local_model = MortgageNetwork(
            args.num_features,
            args.embedding_size,
            args.hidden_dims,
            activation=nn.ReLU(),
            device=torch.device('cpu'),
        )

    # ====================================================================== #
    #  Main Training Loop                                                    #
    # ====================================================================== #

    args.verbose = 1 if myrank == 0 else 0
    args.mysize = mysize
    args.myrank = myrank
    start_time = time.time()
    if gpu_id >= 0:
        # Hogwild Training
        with torch.cuda.device(gpu_id):
            if distributed_hw:
                local_model = local_model.cuda()
                local_model.train()
            for epoch in range(args.epochs):
                train_epoch(
                    model,
                    local_model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    train_sampler,
                    epoch,
                    args,
                )
    else:
        # Horovod, BytePS & Serial
        for epoch in range(args.epochs):
            train_epoch(
                model,
                local_model,
                loss_fn,
                optimizer,
                train_loader,
                train_sampler,
                epoch,
                args,
            )

    # Print final performance summary.
    # For hogwild, other ranks may still be working...
    if args.verbose:
        if args.batched:
            num_iterations = len(train_loader)
        else:
            num_iterations = len(train_loader) * mysize
        ex_per_epoch = num_iterations * allreduce_batch_size
        total_examples = args.epochs * ex_per_epoch
        total_time = time.time() - start_time
        print(
            "\n\t{} Epochs in {} seconds ({} examples per epoch)".format(
                args.epochs, total_time, ex_per_epoch
            )
        )
        print(
            "\tTraining Rate [examples/s]: {}".format(
                total_examples / total_time
                )
        )

import os
from tqdm import tqdm
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
import horovod.torch as hvd


def ensure_shared_grads(model, shared_model, cpu=False):
    for param, shared_param in zip(
        model.parameters(),
        shared_model.parameters()
    ):
        try:
            if cpu:
                shared_param._grad = param.grad.cpu()
            else:
                shared_param._grad = param.grad
        except:
            print('Failed to copy local to shared gradients.')
            raise(RuntimeError)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, optimizer, loader_size, args):
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
    def __init__(self, name, use_hvd):
        self.name = name
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)
        self.use_hvd = use_hvd

    def update(self, val):
        if self.use_hvd:
            self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        else:
            self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


# ========================================================================== #
#                                                                            #
#  Training Function Definition                                              #
#                                                                            #
# ========================================================================== #

def train_epoch(
    model,
    local_model,
    loss_fn,
    optimizer,
    train_loader,
    train_sampler,
    epoch,
    args,
):
    if not args.hogwild:
        model.train()
    if not args.batched:
        train_sampler.set_epoch(epoch)
    train_loss = Metric("train_loss", args.hvd)
    num_iterations = int(len(train_loader) // max(1, args.mysize))
    with tqdm(
        total=num_iterations,
        desc="Train Epoch     #{}".format(epoch + 1),
        disable=not args.verbose,
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == num_iterations:
                break
            if args.hvd and not args.adam:
                adjust_learning_rate(epoch, batch_idx, optimizer, len(train_loader), args)

            if args.hogwild:
                if args.hogwild_gpus > 1:
                    local_model.zero_grad()
                    local_model.load_state_dict(model.state_dict())
                    output = local_model(data.cuda())
                    loss = loss_fn(output, target.cuda())
                    train_loss.update(loss)
                    loss.backward()
                    ensure_shared_grads(local_model, model, cpu=args.cpu_params)
                else:
                    optimizer.zero_grad()
                    output = model(data.cuda())
                    loss = loss_fn(output, target.cuda())
                    train_loss.update(loss)
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
                    train_loss.update(loss)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()

            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({"loss": train_loss.avg.item()})
            t.update(1)


def train(myrank, mysize, model, optimizer, args):

    # ====================================================================== #
    #  Data Handling                                                         #
    # ====================================================================== #

    distributed_hw = False
    if args.hogwild:
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

    local_model = None
    if distributed_hw:
        # Local model definition
        local_model = MortgageNetwork(
            args.num_features,
            args.embedding_size,
            args.hidden_dims,
            activation=nn.ReLU(),
            use_cuda=args.cuda,
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
        # Horovod & Serial
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
            "\n{} Epochs in {} seconds ({} examples per epoch) -> [{} examples/s]".format(
                args.epochs, total_time, ex_per_epoch, total_examples / total_time
            )
        )
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

import horovod.torch as hvd


class SharedAdam(torch_optim.Optimizer):
    """Implements Adam algorithm with shared states.
    BAsed on:
    https://github.com/dgriff777/rl_a3c_pytorch/
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-3,
                 weight_decay=0,
                 amsgrad=False):
        defaults = defaultdict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'].item()
                bias_correction2 = 1 - beta2**state['step'].item()
                step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

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
#  Training Function Definitions                                             #
#                                                                            #
# ========================================================================== #

def train_epoch(model, loss_fn, optimizer, train_loader, train_sampler, epoch, args):
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
            if args.cuda:
                data = data.to("cuda:"+str(args.myrank))
                target = target.to("cuda:"+str(args.myrank))
                #data = data.to("cuda")
                #target = target.to("cuda")
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
    #  Define Loss/Optimization                                              #
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

    # ====================================================================== #
    #  Main Training Loop                                                    #
    # ====================================================================== #

    args.verbose = 1 if myrank == 0 else 0
    args.mysize = mysize
    args.myrank = myrank
    start_time = time.time()
    for epoch in range(args.epochs):
        train_epoch(model, loss_fn, optimizer, train_loader, train_sampler, epoch, args)

    # Print final performance summar,.    # For hogwild, other ranks may still be working...
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

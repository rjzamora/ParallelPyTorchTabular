import batch_dataset, batch_dataloader

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


learning_rate = 0.01
device = "cuda"
num_epochs = 3


def run_training(
    model,
    data_dir,
    batch_size=8096,
    batch_dataload=False,
    num_workers=0,
    use_cuDF=False,
    use_GPU_RAM=False,
):
    # Data
    train_batch_size = batch_size
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
        validation_dataset = batch_dataset_from_parquet(
            os.path.join(out_dir, "validation"),
            batch_size=validation_batch_size,
            use_cuDF=use_cuDF,
            use_GPU_RAM=False,
            num_files=3,
        )
        test_dataset = batch_dataset_from_parquet(
            os.path.join(out_dir, "test"),
            batch_size=validation_batch_size,
            use_cuDF=use_cuDF,
            use_GPU_RAM=False,
            num_files=3,
        )
        train_loader = batch_dataloader.BatchDataLoader(
            train_dataset, shuffle=True
        )
        validation_loader = batch_dataloader.BatchDataLoader(
            validation_dataset, shuffle=False
        )
        test_loader = batch_dataloader.BatchDataLoader(
            test_dataset, shuffle=False
        )
    else:
        train_dataset = dataset_from_parquet(
            os.path.join(out_dir, "train"), epoch_size, shuffle_files=False
        )
        validation_dataset = dataset_from_parquet(
            os.path.join(out_dir, "validation")
        )
        test_dataset = dataset_from_parquet(os.path.join(out_dir, "test"))

        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=train_batch_size, num_workers=num_workers
        )
        validation_loader = torch_data.DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            num_workers=num_workers,
        )
        test_loader = torch_data.DataLoader(
            test_dataset,
            batch_size=validation_batch_size,
            num_workers=num_workers,
        )

    # Optimizer
    optimizer = torch_optim.Adam(model.parameters(), lr=learning_rate)

    # Loss Function
    loss_fn = lambda pred, target: F.binary_cross_entropy_with_logits(
        pred, target
    )

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
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
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # # Test the model
    # model.eval()
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for data, labels in test_loader:
    #         data = data.to(device)
    #         labels = labels.to(device)
    #         outputs = model(data)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     print('Accuracy of the model on the test data: {} %'.format(100 * correct / total))
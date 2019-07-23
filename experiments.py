#!/usr/bin/python

from contextlib import redirect_stdout
import os
import subprocess as sp
import sys
import time
import datetime

""" Run simple timing experiments.
"""

output_dir = "/home/nfs/rzamora/workspace/rapids-dl/ParallelPyTorchTabular/results/"
batch_sizes = [1000, 8000, 64000, 128000]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

common = [
    "python",
    "main.py",
    "--batch_size",
    "512",
    "--batched",
    "--lr",
    "0.01",
    "--par",
    "hog",
    "--hogwild-gpus",
    "1",
    "--adam",
    "--epochs", "1"
]

commands = [
    common+["--hogwild-nprocs", "1"],
    common+["--hogwild-nprocs", "2"],
    common+["--hogwild-nprocs", "3"],
    common+["--hogwild-nprocs", "4"],
]

if __name__ == "__main__":

    dt = datetime.datetime.today()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    file_tag = dt.strftime("%d-%B-%Y-%H-%M-%S")
    out_name = output_dir + 'output.' + file_tag

    exp_cnt = 0
    with open(out_name, 'w') as outf:
        with redirect_stdout(outf):

            print("ParallelPyTorchTabular Experiments")
            print("==================================")
            print("Current date and time: ", dt,"\n")
            sys.stdout.flush()

            for run_cmd in commands:

                print("\nExperiment Number ", exp_cnt + 1, "\n")
                print("\tcommand:", *run_cmd)
                print("\n")
                sys.stdout.flush()

                sp.call(run_cmd, stdout=outf, stderr=outf)
                exp_cnt += 1



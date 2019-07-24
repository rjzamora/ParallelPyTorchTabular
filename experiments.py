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
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

common = [
    "--batch-size",
    "100000",
    "--batched",
    "--lr",
    "0.01",
    "--adam",
    "--epochs",
    "10"
]

common_adam = [
    "--adam",
    "--batched",
    "--epochs",
    "10"
]

common_sgd = [
    "--batched",
    "--epochs",
    "10"
]

commands = [
    # Grid search over batch-size and lr for Adam
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.008","--batch-size","1000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.008","--batch-size","10000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.008","--batch-size","100000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.010","--batch-size","1000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.010","--batch-size","10000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.010","--batch-size","100000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.012","--batch-size","1000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.012","--batch-size","10000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.012","--batch-size","100000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.015","--batch-size","1000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.015","--batch-size","10000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.015","--batch-size","100000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.020","--batch-size","1000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.020","--batch-size","10000","--momentum","0","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_adam+["--lr","0.020","--batch-size","100000","--momentum","0","--wd","0"], 1),

    # Grid search over batch-size and lr for basic SGD
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.015","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.015","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.015","--batch-size","100000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","100000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.025","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.025","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.025","--batch-size","100000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.030","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.030","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.030","--batch-size","100000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.035","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.035","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.035","--batch-size","100000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.040","--batch-size","1000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.040","--batch-size","10000","--momentum","0.9","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.040","--batch-size","100000","--momentum","0.9","--wd","0"], 1),

    # Test sensitivity to momentum
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.88","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.89","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.90","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.91","--wd","0"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common_sgd+["--lr","0.020","--batch-size","10000","--momentum","0.92","--wd","0"], 1),

    # Horovod scaling with Adam (TTS not likely to scale with Adam?)
    ("1,2,3,4,5,7", ["mpirun", "-n", "1","python","main.py"]+common+["--par","hvd"], 1),
    ("1,2,3,4,5,7", ["mpirun", "-n", "2","python","main.py"]+common+["--par","hvd"], 1),
    ("1,2,3,4,5,7", ["mpirun", "-n", "3","python","main.py"]+common+["--par","hvd"], 1),
    ("1,2,3,4,5,7", ["mpirun", "-n", "4","python","main.py"]+common+["--par","hvd"], 1),
    ("1,2,3,4,5,7", ["mpirun", "-n", "5","python","main.py"]+common+["--par","hvd"], 1),
    ("1,2,3,4,5,7", ["mpirun", "-n", "6","python","main.py"]+common+["--par","hvd"], 1),

    # Distributed Hogwild scaling with Adam (TTS not likely to scale with Adam?)
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","1","--hogwild-gpus","1","--gpu-params"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","2","--hogwild-gpus","2","--gpu-params"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","3","--hogwild-gpus","3","--gpu-params"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","4","--hogwild-gpus","4","--gpu-params"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","5","--hogwild-gpus","5","--gpu-params"], 1),
    ("1,2,3,4,5,7", ["python","main.py"]+common+["--par","hog","--hogwild-procs","6","--hogwild-gpus","6","--gpu-params"], 1),

    # BytePS scaling with Adam (TTS not likely to scale with Adam?)
    ("1", ["python","launch_bps.py"]+common+["--par","bps"], 1),
    ("1,2", ["python","launch_bps.py"]+common+["--par","bps"], 1),
    ("1,2,3", ["python","launch_bps.py"]+common+["--par","bps"], 1),
    ("1,2,3,4", ["python","launch_bps.py"]+common+["--par","bps"], 1),
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

            for (cvd, run_cmd, ntrials) in commands:

                print("\nExperiment Number ", exp_cnt + 1, "\n")
                print("\tcommand:", *run_cmd)
                print("\n")
                sys.stdout.flush()

                for trial in range(ntrials):
                    os.environ["CUDA_VISIBLE_DEVICES"] = cvd
                    print("\tTrial ", trial+1, "...\n")
                    print("\tCUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
                    sys.stdout.flush()

                    sp.call(run_cmd, stdout=outf, stderr=outf)

                exp_cnt += 1

#!/usr/bin/python

from __future__ import print_function
import os
import subprocess
import threading
import sys
import time

""" This script can be used to launch a single-machine BytePS job
"""

timeout = 60 * 10 # Timeout == 10 mins.

def worker(local_rank, local_size, command):
    my_env = os.environ.copy()
    my_env["BYTEPS_LOCAL_RANK"] = str(local_rank)
    my_env["BYTEPS_LOCAL_SIZE"] = str(local_size)
    if command.find("python") != 0:
        command = "python main.py " + command
    subprocess.check_call(
        command, env=my_env, stdout=sys.stdout, stderr=sys.stderr, shell=True
    )


if __name__ == "__main__":
    print("BytePS launching worker")
    sys.stdout.flush()

    os.environ["DMLC_WORKER_ID"]="0" # your worker id
    os.environ["DMLC_NUM_WORKER"]="1" # you only have one worker
    os.environ["DMLC_ROLE"]="worker" # your role is worker
    os.environ["DMLC_ENABLE_RDMA"]="1" # Enable RDMA

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    local_size = len(devices.split(","))
    t = [None] * local_size
    for i in range(local_size):
        command = " ".join(sys.argv[1:])
        t[i] = threading.Thread(target=worker, args=[i, local_size, command])
        t[i].daemon = True
        t[i].start()

    for i in range(local_size):
        t[i].join(timeout=timeout)

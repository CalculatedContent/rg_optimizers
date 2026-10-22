"""Configure native TorchTPU topology in the parent, then spawn eight workers."""
import os
from pathlib import Path
import subprocess
import socket
import sys


def main():
    import torch
    import torch_tpu  # noqa: F401
    # Official launch helper must run before any TPU tensor or process group.
    torch.tpu.distributed.set_tpu_launch_env(nnodes=1)
    # The 0.2.0 Python helper sets topology/addresses; only its CLI prints
    # WORLD_SIZE. Derive the count from the actual detected device addresses.
    addresses = os.environ.get('TORCH_TPU_SLICEBUILDER_ADDRESSES', '')
    count = len([x for x in addresses.split(',') if x])
    if count != 8 or int(os.environ.get('WORLD_SIZE', '8')) != 8:
        raise RuntimeError('This baseline requires an existing eight-device slice')
    os.environ['WORLD_SIZE'] = '8'
    if int(os.environ.get('NODE_RANK', '0')) != 0:
        raise RuntimeError('This launcher is for the existing single-host v5e-8 only')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1',
           f'--master-port={port}',
           '--nnodes=1', '--nproc-per-node=8', '--max-restarts=0',
           str(Path(__file__).with_name('train.py')), '--expected-world=8', *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()

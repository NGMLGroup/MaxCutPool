import os
import torch
import numpy as np
import subprocess as sp
from typing import Tuple


def byte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(value / (1024 * 1024), digits)

def medibyte_to_megabyte(value: int, digits: int = 2) -> float:
    return round(1.0485 * value, digits)

def get_gpu_memory_from_nvidia_smi(  # pragma: no cover
    device: int = 0,
    digits: int = 2,
) -> Tuple[float, float]:
    r"""Returns the free and used GPU memory in megabytes, as reported by
    :obj:`nivdia-smi`.

    .. note::

        :obj:`nvidia-smi` will generally overestimate the amount of memory used
        by the actual program, see `here <https://pytorch.org/docs/stable/
        notes/faq.html#my-gpu-memory-isn-t-freed-properly>`__.

    Args:
        device (int, optional): The GPU device identifier. (default: :obj:`1`)
        digits (int): The number of decimals to use for megabytes.
            (default: :obj:`2`)
    """
    CMD = 'nvidia-smi --query-gpu=memory.free --format=csv'
    free_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    CMD = 'nvidia-smi --query-gpu=memory.used --format=csv'
    used_out = sp.check_output(CMD.split()).decode('utf-8').split('\n')[1:-1]

    if device < 0 or device >= len(free_out):
        raise AttributeError(
            f'GPU {device} not available (found {len(free_out)} GPUs)')

    free_mem = medibyte_to_megabyte(int(free_out[device].split()[0]), digits)
    used_mem = medibyte_to_megabyte(int(used_out[device].split()[0]), digits)

    return free_mem, used_mem

def find_devices(max_devices: int = 1, greedy: bool = False, gamma: int = 12):
    # if no gpus are available return None
    if not torch.cuda.is_available():
        return max_devices
    n_gpus = torch.cuda.device_count()
    # if only 1 gpu, return 1 (i.e., the number of devices)
    if n_gpus == 1:
        return 1
    # if multiple gpus are available, return gpu id list with length max_devices
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible_devices is not None:
        visible_devices = [int(i) for i in visible_devices.split(',')]
    else:
        visible_devices = range(n_gpus)
    available_memory = np.asarray([get_gpu_memory_from_nvidia_smi(device)[0]
                                   for device in visible_devices])
    # if greedy, return `max_devices` gpus sorted by available capacity
    if greedy:
        devices = np.argsort(available_memory)[::-1].tolist()
        return devices[:max_devices]
    # otherwise sample `max_devices` gpus according to available capacity
    p = (available_memory / np.linalg.norm(available_memory, gamma)) ** gamma
    # ensure p sums to 1
    p = p / p.sum()
    devices = np.random.choice(np.arange(len(p)), size=max_devices,
                               replace=False, p=p)
    return devices.tolist()

def reduce_precision():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
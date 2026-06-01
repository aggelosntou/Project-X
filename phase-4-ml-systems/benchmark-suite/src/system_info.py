"""
system_info.py
==============
Collect hardware and software information about the current machine.
Works on macOS and Linux; silently degrades on unsupported platforms.
"""

import platform
import os
import subprocess
import sys


def _get_cpu_brand() -> str:
    """Return CPU brand string from sysctl (macOS) or /proc/cpuinfo (Linux)."""
    # macOS
    try:
        return subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass
    # Linux
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return 'Unknown'


def _get_cache_sizes() -> dict:
    """Return L1/L2/L3 cache sizes in KB."""
    result = {'l1_cache_kb': 'N/A', 'l2_cache_kb': 'N/A', 'l3_cache_kb': 'N/A'}
    # macOS: three separate sysctl calls
    try:
        keys = ['hw.l1dcachesize', 'hw.l2cachesize', 'hw.l3cachesize']
        label_map = ['l1_cache_kb', 'l2_cache_kb', 'l3_cache_kb']
        for key, label in zip(keys, label_map):
            try:
                out = subprocess.check_output(
                    ['sysctl', '-n', key], stderr=subprocess.DEVNULL
                ).decode().strip()
                if out:
                    result[label] = int(out) // 1024
            except Exception:
                pass
        return result
    except Exception:
        pass
    # Linux: read from /sys
    try:
        base = '/sys/devices/system/cpu/cpu0/cache'
        if os.path.isdir(base):
            for entry in sorted(os.listdir(base)):
                level_file = os.path.join(base, entry, 'level')
                type_file  = os.path.join(base, entry, 'type')
                size_file  = os.path.join(base, entry, 'size')
                if not all(os.path.exists(f) for f in [level_file, type_file, size_file]):
                    continue
                with open(level_file) as f: level = f.read().strip()
                with open(type_file)  as f: ctype = f.read().strip()
                with open(size_file)  as f: size  = f.read().strip()
                if ctype == 'Instruction':
                    continue
                label_map = {'1': 'l1_cache_kb', '2': 'l2_cache_kb', '3': 'l3_cache_kb'}
                if level in label_map:
                    # size is like "32K"
                    sz = size.rstrip('K').rstrip('k')
                    result[label_map[level]] = int(sz) if sz.isdigit() else size
    except Exception:
        pass
    return result


def get_system_info() -> dict:
    """Return a dict with OS, CPU, memory, and library version information."""
    info: dict = {}

    info['os']                = platform.system() + ' ' + platform.release()
    info['python']            = platform.python_version()
    info['cpu']               = platform.processor() or _get_cpu_brand()
    info['cpu_arch']          = platform.machine()

    # psutil is optional — gracefully degrade
    try:
        import psutil
        info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
        info['cpu_cores_logical']  = psutil.cpu_count(logical=True)
        info['ram_gb']             = round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except ImportError:
        info['cpu_cores_physical'] = os.cpu_count() or 'N/A'
        info['cpu_cores_logical']  = os.cpu_count() or 'N/A'
        info['ram_gb']             = 'N/A (install psutil)'

    info.update(_get_cache_sizes())

    # Library versions
    try:
        import numpy as np
        info['numpy_version'] = np.__version__
    except ImportError:
        info['numpy_version'] = 'not installed'

    return info


def print_system_info() -> None:
    """Pretty-print system information to stdout."""
    info = get_system_info()
    print("=" * 50)
    print("System Information")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k:<28} {v}")
    print()


if __name__ == '__main__':
    print_system_info()

"""Import before genesis so its CUDA backend finds the GPU driver on WSL2.

Genesis loads "libcuda.so", which on WSL2 only exists in /usr/lib/wsl/lib, a directory the
loader does not search by default. The loader reads LD_LIBRARY_PATH once at startup, so
setting it here is not enough: the script restarts itself with the path added.
"""
import os
import sys

WSL_LIB = "/usr/lib/wsl/lib"

paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
if os.path.isdir(WSL_LIB) and WSL_LIB not in paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join([WSL_LIB] + paths)
    os.execv(sys.executable, sys.orig_argv)

===============
GPU on Windows
===============

There are two ways to accelerate training with a GPU on Windows:

- ``device_type=gpu`` (OpenCL): pre-built into the official Python packages since
  version ``3.2.0``, so ``pip install lightgbm`` is enough. This is the easiest
  option and works on any GPU vendor providing OpenCL 1.2+.
- ``device_type=cuda``: a separate implementation that puts the whole training
  process on NVIDIA GPUs. On Windows it must be built from source (see the
  `Installation Guide <./Installation-Guide.rst#build-cuda-version>`__).

CUDA version on Windows
=======================

Build with **Visual Studio 2022 (or newer)**, **CMake** and the **CUDA Toolkit**
(the same toolchain as for the Linux build, see
`Installation Guide <./Installation-Guide.rst#build-cuda-version>`__ for details).

Notes:

- The build automatically falls back to a bundled NCCL stub, because NCCL has no
  Windows distribution. This is sufficient for single-GPU training (``num_gpu=1``,
  the default): every NCCL call site is only reached when ``num_gpu > 1``.
- Attempting ``num_gpu > 1`` against the stub aborts with a clear error message
  instead of silently producing wrong results.
- Compute-capability support follows the CUDA Toolkit version used for building
  (see ``CMAKE_CUDA_ARCHITECTURES`` in the installation guide).

Multi-GPU (``num_gpu > 1``), experimental
=========================================

The multi-GPU path (``NCCLGBDT``) can be reached on Windows by building a real
NCCL and pointing CMake at it:

- NVIDIA's NCCL source tree ships a Windows OS layer since version 2.32, so NCCL
  can be built from source with MSVC (see the `NCCL repository
  <https://github.com/NVIDIA/nccl>`__; community build recipes exist, e.g.
  https://github.com/WyvernTKC/nccl-windows, which tracks NVIDIA's official
  releases plus a small Windows build recipe).
- Configure LightGBM with ``-DNCCL_ROOT=<nccl install prefix>``; on Windows the
  shared NCCL (``nccl.lib`` / ``nccl.dll``) is used by default
  (``BUILD_WITH_SHARED_NCCL``).

Status: validated end-to-end on a single GPU (the whole ``num_gpu > 1`` code path
runs through a single-rank NCCL communicator and matches the CPU learner at the
default ``max_bin``), but the **cross-GPU transport (2+ GPUs) is still experimental
and needs broader hardware testing**. Known limitation inherited from the upstream
multi-GPU implementation: bagging is not supported.

Further Reading
---------------

- `GPU Tutorial <./GPU-Tutorial.rst>`__
- `GPU Tuning Guide and Performance Comparison <./GPU-Performance.rst>`__
- `GPU SDK Correspondence and Device Targeting Table <./GPU-Targets.rst>`__

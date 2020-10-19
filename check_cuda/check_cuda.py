#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Outputs some information on CUDA-enabled devices on your computer,
including current memory usage.

It's a port of https://gist.github.com/f0k/0d6431e3faa60bffc788f8b4daa029b1
from C to Python with ctypes, so it can run without compiling anything. Note
that this is a direct translation with no attempt to make the code Pythonic.
It's meant as a general demonstration on how to obtain CUDA device information
from Python without resorting to nvidia-smi or a compiled Python extension.

Author: Jan Schlüter
"""

import ctypes
import logging
import os
import sys
from typing import List

import yaml
from singleton_decorator import singleton

from .data_models.device import Device

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

LOGGER = logging.getLogger(__name__)


def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {
        (1, 0): 8,  # Tesla
        (1, 1): 8,
        (1, 2): 8,
        (1, 3): 8,
        (2, 0): 32,  # Fermi
        (2, 1): 48,
        (3, 0): 192,  # Kepler
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,  # Maxwell
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,  # Pascal
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,  # Volta
        (7, 2): 64,
        (7, 5): 64,  # Turing
    }.get((major, minor), 0)


@singleton
class CheckCuda(object):
    # __instance = None

    # #__is_already_load = False

    # def __new__(cls):
    #     if CheckCuda.__instance is None:
    #         CheckCuda.__instance = object.__new__(cls)
    #     return CheckCuda.__instance

    def __init__(self):
        self.__cuda = None
        self.__cuda_device_list = []

    def get_cuda_info(self) -> List[Device]:
        if self.__cuda is None:
            libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
            for libname in libnames:
                try:
                    self.__cuda = ctypes.CDLL(libname)
                    LOGGER.info('Loading cuda libraries')
                except OSError:
                    continue
                else:
                    break
            if self.__cuda is not None:

                nGpus = ctypes.c_int()
                name = b' ' * 100
                cc_major = ctypes.c_int()
                cc_minor = ctypes.c_int()
                cores = ctypes.c_int()
                threads_per_core = ctypes.c_int()
                clockrate = ctypes.c_int()
                freeMem = ctypes.c_size_t()
                totalMem = ctypes.c_size_t()

                result = ctypes.c_int()
                device = ctypes.c_int()
                context = ctypes.c_void_p()
                error_str = ctypes.c_char_p()
                is_continue = True
                while is_continue:
                    is_continue = False
                    result = self.__cuda.cuInit(0)
                    if result != CUDA_SUCCESS:
                        self.__cuda.cuGetErrorString(result,
                                                     ctypes.byref(error_str))
                        LOGGER.error("cuInit failed with error code %d: %s" %
                                     (result, error_str.value.decode()))
                        break
                    result = self.__cuda.cuDeviceGetCount(ctypes.byref(nGpus))
                    if result != CUDA_SUCCESS:
                        self.__cuda.cuGetErrorString(result,
                                                     ctypes.byref(error_str))
                        LOGGER.error(
                            "cuDeviceGetCount failed with error code %d: %s" %
                            (result, error_str.value.decode()))
                        break
                    LOGGER.debug("Found %d device(s)." % nGpus.value)
                    for i in range(nGpus.value):
                        cuda_device_name = ''
                        cuda_compute_capability_major = 0
                        cuda_compute_capability_minor = 0
                        cuda_cores = 0
                        cuda_concurrent_threads = 0
                        cuda_gpu_clock_mhz = 0
                        cuda_memory_clock_mhz = 0
                        cuda_total_memory_mib = 0
                        cuda_free_memory_mib = 0

                        result = self.__cuda.cuDeviceGet(
                            ctypes.byref(device), i)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(
                                result, ctypes.byref(error_str))
                            LOGGER.error(
                                "cuDeviceGet failed with error code %d: %s" %
                                (result, error_str.value.decode()))
                            break
                        LOGGER.debug("Nvidia Device: %d" % i)

                        if self.__cuda.cuDeviceGetName(ctypes.c_char_p(name),
                                                       len(name),
                                                       device) == CUDA_SUCCESS:
                            cuda_device_name = (name.split(b'\0',
                                                           1)[0].decode())

                            LOGGER.debug("  Name: %s" % cuda_device_name)
                        if self.__cuda.cuDeviceComputeCapability(
                                ctypes.byref(cc_major), ctypes.byref(cc_minor),
                                device) == CUDA_SUCCESS:
                            cuda_compute_capability_major = cc_major.value
                            cuda_compute_capability_minor = cc_minor.value

                            LOGGER.debug("  Compute Capability: %d.%d" %
                                        (cuda_compute_capability_major,
                                         cuda_compute_capability_minor))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(cores),
                                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                device) == CUDA_SUCCESS:
                            LOGGER.debug("  Multiprocessors: %d" % cores.value)
                            cuda_cores = cores.value * ConvertSMVer2Cores(
                                cc_major.value, cc_minor.value)
                            LOGGER.debug("  CUDA Cores: %s" %
                                        (cuda_cores or "unknown"))
                            if self.__cuda.cuDeviceGetAttribute(
                                    ctypes.byref(threads_per_core),
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                    device) == CUDA_SUCCESS:
                                cuda_concurrent_threads = cores.value * threads_per_core.value
                                LOGGER.debug("  Concurrent threads: %d" %
                                            (cuda_concurrent_threads))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(clockrate),
                                CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                device) == CUDA_SUCCESS:
                            cuda_gpu_clock_mhz = clockrate.value / 1000.
                            LOGGER.debug("  GPU clock: %g MHz" %
                                        (cuda_gpu_clock_mhz))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(clockrate),
                                CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                device) == CUDA_SUCCESS:
                            cuda_memory_clock_mhz = clockrate.value / 1000.
                            LOGGER.debug("  Memory clock: %g MHz" %
                                        (cuda_memory_clock_mhz))
                        result = self.__cuda.cuCtxCreate(
                            ctypes.byref(context), 0, device)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(
                                result, ctypes.byref(error_str))
                            LOGGER.error(
                                "cuCtxCreate failed with error code %d: %s" %
                                (result, error_str.value.decode()))
                        else:
                            result = self.__cuda.cuMemGetInfo(
                                ctypes.byref(freeMem), ctypes.byref(totalMem))
                            if result == CUDA_SUCCESS:
                                cuda_total_memory_mib = totalMem.value / 1024**2
                                LOGGER.debug("  Total Memory: %ld MiB" %
                                            (cuda_total_memory_mib))

                                cuda_free_memory_mib = freeMem.value / 1024**2

                                LOGGER.debug("  Free Memory: %ld MiB" %
                                      (cuda_free_memory_mib))
                            else:
                                self.__cuda.cuGetErrorString(
                                    result, ctypes.byref(error_str))
                                LOGGER.error(
                                    "cuMemGetInfo failed with error code %d: %s"
                                    % (result, error_str.value.decode()))
                            self.__cuda.cuCtxDetach(context)
                        self.__cuda_device_list.append(
                            Device(
                                cuda_device_name,
                                cuda_compute_capability_major,
                                cuda_compute_capability_minor,
                                cuda_cores,
                                cuda_concurrent_threads,
                                cuda_gpu_clock_mhz,
                                cuda_memory_clock_mhz,
                                cuda_total_memory_mib,
                                cuda_free_memory_mib,
                            ))

        return self.__cuda_device_list

    def is_cuda_available(self) -> bool:
        return len(self.get_cuda_info()) > 0


def is_cuda_available() -> bool:
    return CheckCuda().is_cuda_available()


def get_cuda_info() -> List[Device]:
    return CheckCuda().get_cuda_info()

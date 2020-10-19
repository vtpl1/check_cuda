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

import sys
import ctypes
import logging
import yaml
import os
from singleton_decorator import singleton
from .data_models.device import Device
from typing import List
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
                    LOGGER.info('Loading libraries')
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
                    LOGGER.error("Found %d device(s)." % nGpus.value)
                    for i in range(nGpus.value):
                        result = self.__cuda.cuDeviceGet(
                            ctypes.byref(device), i)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(
                                result, ctypes.byref(error_str))
                            LOGGER.error("cuDeviceGet failed with error code %d: %s" %
                                  (result, error_str.value.decode()))
                            break
                        LOGGER.info("Nvidia Device: %d" % i)
                        cuda_device_name = ''
                        if self.__cuda.cuDeviceGetName(ctypes.c_char_p(name),
                                                       len(name),
                                                       device) == CUDA_SUCCESS:
                            cuda_device_name = (name.split(b'\0', 1)[0].decode())

                            LOGGER.info("  Name: %s" %
                                  cuda_device_name)
                        if self.__cuda.cuDeviceComputeCapability(
                                ctypes.byref(cc_major), ctypes.byref(cc_minor),
                                device) == CUDA_SUCCESS:
                            print("  Compute Capability: %d.%d" %
                                  (cc_major.value, cc_minor.value))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(cores),
                                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                device) == CUDA_SUCCESS:
                            print("  Multiprocessors: %d" % cores.value)
                            print("  CUDA Cores: %s" %
                                  (cores.value * ConvertSMVer2Cores(
                                      cc_major.value, cc_minor.value)
                                   or "unknown"))
                            if self.__cuda.cuDeviceGetAttribute(
                                    ctypes.byref(threads_per_core),
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                    device) == CUDA_SUCCESS:
                                print("  Concurrent threads: %d" %
                                      (cores.value * threads_per_core.value))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(clockrate),
                                CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                device) == CUDA_SUCCESS:
                            print("  GPU clock: %g MHz" %
                                  (clockrate.value / 1000.))
                        if self.__cuda.cuDeviceGetAttribute(
                                ctypes.byref(clockrate),
                                CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                device) == CUDA_SUCCESS:
                            print("  Memory clock: %g MHz" %
                                  (clockrate.value / 1000.))
                        result = self.__cuda.cuCtxCreate(
                            ctypes.byref(context), 0, device)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(
                                result, ctypes.byref(error_str))
                            print("cuCtxCreate failed with error code %d: %s" %
                                  (result, error_str.value.decode()))
                        else:
                            result = self.__cuda.cuMemGetInfo(
                                ctypes.byref(freeMem), ctypes.byref(totalMem))
                            if result == CUDA_SUCCESS:
                                print("  Total Memory: %ld MiB" %
                                      (totalMem.value / 1024**2))
                                print("  Free Memory: %ld MiB" %
                                      (freeMem.value / 1024**2))
                            else:
                                self.__cuda.cuGetErrorString(
                                    result, ctypes.byref(error_str))
                                print(
                                    "cuMemGetInfo failed with error code %d: %s"
                                    % (result, error_str.value.decode()))
                            self.__cuda.cuCtxDetach(context)
                        self.__is_cuda_available = True

        return self.__cuda_device_list

    def is_cuda_available(self) -> bool:
        return len(self.get_cuda_info()) > 0

def setup_logging(default_path='logging.yaml',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup logging configuration

    """

    path = default_path
    print("Current working directory", path)
    # value = os.getenv(env_key, None)
    # if value:
    #     path = value
    if os.path.exists(path):
        print("Got path")
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        print("Not got path")
        logging.basicConfig(level=default_level,
                            format="%(levelname)s - %(name)45s - %(message)s")


def main():
    from .get_hardware_info import get_hardware_info
    setup_logging()
    get_hardware_info()
    # from cpuinfo import get_cpu_info
    # x = get_cpu_info()
    # print(x, "------------------------------------")
    # print(x['brand_raw'])
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
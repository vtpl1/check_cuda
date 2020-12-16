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
import pynvml as N
import psutil
import time
from pprint import pprint

import yaml
from singleton_decorator import singleton

from .data_models.cuda_device import CudaDevice

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024

LOGGER = logging.getLogger(__name__)


def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {
        (1, 0): 8,    # Tesla
        (1, 1): 8,
        (1, 2): 8,
        (1, 3): 8,
        (2, 0): 32,    # Fermi
        (2, 1): 48,
        (3, 0): 192,    # Kepler
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,    # Maxwell
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,    # Pascal
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,    # Volta
        (7, 2): 64,
        (7, 5): 64,    # Turing
    }.get((major, minor), 0)


@singleton
class CheckCuda(object):
    def __init__(self):
        self.__cuda = None
        self.__cuda_device_list = {}
        # self.__processes = {}
        print("Starting NVML")
        N.nvmlInit()

    def _decode(self, b):
        if isinstance(b, bytes):
            return b.decode('utf-8')    # for python3, to unicode
        return b

    def __del__(self):
        print("Shutting down NVML")
        N.nvmlShutdown()

    def get_process_info(self, nv_process):
        process = {}
        ps_process = psutil.Process(pid=nv_process.pid)
        process['username'] = ps_process.username()
        # cmdline returns full path;
        # as in `ps -o comm`, get short cmdnames.
        _cmdline = ps_process.cmdline()
        if not _cmdline:
            # sometimes, zombie or unknown (e.g. [kworker/8:2H])
            process['command'] = '?'
            process['full_command'] = ['?']
        else:
            process['command'] = os.path.basename(_cmdline[0])
            process['full_command'] = _cmdline
        # Bytes to MBytes
        # if drivers are not TTC this will be None.
        usedmem = nv_process.usedGpuMemory // MB if \
                    nv_process.usedGpuMemory else None
        process['gpu_memory_usage'] = usedmem
        process['cpu_percent'] = ps_process.cpu_percent()
        process['cpu_memory_usage'] = \
            round((ps_process.memory_percent() / 100.0) *
                    psutil.virtual_memory().total)
        process['pid'] = nv_process.pid
        pprint(process)
        return process

    def get_gpu_info(self, index) -> List[CudaDevice]:
        """Get one GPU information specified by nvml handle"""
        handle = N.nvmlDeviceGetHandleByIndex(index)
        name = self._decode(N.nvmlDeviceGetName(handle))
        uuid = self._decode(N.nvmlDeviceGetUUID(handle))

        try:
            temperature = N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)
        except N.NVMLError:
            temperature = None    # Not supported
        try:
            fan_speed = N.nvmlDeviceGetFanSpeed(handle)
        except N.NVMLError:
            fan_speed = None    # Not supported
        try:
            memory = N.nvmlDeviceGetMemoryInfo(handle)    # in Bytes
        except N.NVMLError:
            memory = None    # Not supported

        try:
            utilization = N.nvmlDeviceGetUtilizationRates(handle)
        except N.NVMLError:
            utilization = None    # Not supported

        try:
            utilization_enc = N.nvmlDeviceGetEncoderUtilization(handle)
        except N.NVMLError:
            utilization_enc = None    # Not supported

        try:
            utilization_dec = N.nvmlDeviceGetDecoderUtilization(handle)
        except N.NVMLError:
            utilization_dec = None    # Not supported

        try:
            power = N.nvmlDeviceGetPowerUsage(handle)
        except N.NVMLError:
            power = None

        try:
            power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
        except N.NVMLError:
            power_limit = None

        try:
            nv_comp_processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
        except N.NVMLError:
            nv_comp_processes = None    # Not supported
        try:
            nv_graphics_processes = N.nvmlDeviceGetGraphicsRunningProcesses(handle)
        except N.NVMLError:
            nv_graphics_processes = None    # Not supported

        if nv_comp_processes is None and nv_graphics_processes is None:
            processes = None
        else:
            processes = []
            nv_comp_processes = nv_comp_processes or []
            nv_graphics_processes = nv_graphics_processes or []
            # A single process might run in both of graphics and compute mode,
            # However we will display the process only once
            seen_pids = set()
            for nv_process in nv_comp_processes + nv_graphics_processes:
                if nv_process.pid in seen_pids:
                    continue
                seen_pids.add(nv_process.pid)
                try:
                    process = self.get_process_info(nv_process)
                    processes.append(process)
                except psutil.NoSuchProcess:
                    # TODO: add some reminder for NVML broken context
                    # e.g. nvidia-smi reset  or  reboot the system
                    pass

            # TODO: Do not block if full process info is not requested
            # time.sleep(0.1)
            # for process in processes:
            #     pid = process['pid']
            #     cache_process = self.__processes[pid]
            #     process['cpu_percent'] = cache_process.cpu_percent()
        gpu_info = {
            'index': index,
            'uuid': uuid,
            'name': name,
            'temperature.gpu': temperature,
            'fan.speed': fan_speed,
            'utilization.gpu': utilization.gpu if utilization else None,
            'utilization.enc': utilization_enc[0] if utilization_enc else None,
            'utilization.dec': utilization_dec[0] if utilization_dec else None,
            'power.draw': power // 1000 if power is not None else None,
            'enforced.power.limit': power_limit // 1000 if power_limit is not None else None,
        # Convert bytes into MBytes
            'memory.used': memory.used // MB if memory else None,
            'memory.total': memory.total // MB if memory else None,
            'processes': processes,
        }
        pprint(gpu_info)
        return self.__cuda_device_list

    def get_cuda_info(self) -> List[CudaDevice]:
        # 1. get list of gpu
        gpu_list = []
        device_count = N.nvmlDeviceGetCount()
        for index in range(device_count):
            gpu_info = self.get_gpu_info(index)
            gpu_list.append(gpu_info)
        return self.__cuda_device_list

    def get_cuda_info1(self) -> List[CudaDevice]:
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
                        self.__cuda.cuGetErrorString(result, ctypes.byref(error_str))
                        LOGGER.error("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
                        break
                    result = self.__cuda.cuDeviceGetCount(ctypes.byref(nGpus))
                    if result != CUDA_SUCCESS:
                        self.__cuda.cuGetErrorString(result, ctypes.byref(error_str))
                        LOGGER.error("cuDeviceGetCount failed with error code %d: %s" %
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

                        result = self.__cuda.cuDeviceGet(ctypes.byref(device), i)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(result, ctypes.byref(error_str))
                            LOGGER.error("cuDeviceGet failed with error code %d: %s" %
                                         (result, error_str.value.decode()))
                            break
                        LOGGER.debug("Nvidia Device: %d" % i)

                        if self.__cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
                            cuda_device_name = (name.split(b'\0', 1)[0].decode())

                            LOGGER.debug("  Name: %s" % cuda_device_name)
                        if self.__cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor),
                                                                 device) == CUDA_SUCCESS:
                            cuda_compute_capability_major = cc_major.value
                            cuda_compute_capability_minor = cc_minor.value

                            LOGGER.debug("  Compute Capability: %d.%d" %
                                         (cuda_compute_capability_major, cuda_compute_capability_minor))
                        if self.__cuda.cuDeviceGetAttribute(ctypes.byref(cores),
                                                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                                            device) == CUDA_SUCCESS:
                            LOGGER.debug("  Multiprocessors: %d" % cores.value)
                            cuda_cores = cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value)
                            LOGGER.debug("  CUDA Cores: %s" % (cuda_cores or "unknown"))
                            if self.__cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core),
                                                                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                                                device) == CUDA_SUCCESS:
                                cuda_concurrent_threads = cores.value * threads_per_core.value
                                LOGGER.debug("  Concurrent threads: %d" % (cuda_concurrent_threads))
                        if self.__cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                                            device) == CUDA_SUCCESS:
                            cuda_gpu_clock_mhz = clockrate.value / 1000.
                            LOGGER.debug("  GPU clock: %g MHz" % (cuda_gpu_clock_mhz))
                        if self.__cuda.cuDeviceGetAttribute(ctypes.byref(clockrate),
                                                            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                                            device) == CUDA_SUCCESS:
                            cuda_memory_clock_mhz = clockrate.value / 1000.
                            LOGGER.debug("  Memory clock: %g MHz" % (cuda_memory_clock_mhz))
                        result = self.__cuda.cuCtxCreate(ctypes.byref(context), 0, device)
                        if result != CUDA_SUCCESS:
                            self.__cuda.cuGetErrorString(result, ctypes.byref(error_str))
                            LOGGER.error("cuCtxCreate failed with error code %d: %s" %
                                         (result, error_str.value.decode()))
                        else:
                            result = self.__cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
                            if result == CUDA_SUCCESS:
                                cuda_total_memory_mib = totalMem.value / 1024**2
                                LOGGER.debug("  Total Memory: %ld MiB" % (cuda_total_memory_mib))

                                cuda_free_memory_mib = freeMem.value / 1024**2

                                LOGGER.debug("  Free Memory: %ld MiB" % (cuda_free_memory_mib))
                            else:
                                self.__cuda.cuGetErrorString(result, ctypes.byref(error_str))
                                LOGGER.error("cuMemGetInfo failed with error code %d: %s" %
                                             (result, error_str.value.decode()))
                            self.__cuda.cuCtxDetach(context)
                        self.__cuda_device_list[i] = CudaDevice(
                            i,
                            cuda_device_name,
                            cuda_compute_capability_major,
                            cuda_compute_capability_minor,
                            cuda_cores,
                            cuda_concurrent_threads,
                            cuda_gpu_clock_mhz,
                            cuda_memory_clock_mhz,
                            cuda_total_memory_mib,
                            cuda_free_memory_mib,
                        )

        return self.__cuda_device_list

    def is_cuda_available(self) -> bool:
        return len(self.get_cuda_info()) > 0


def is_cuda_available() -> bool:
    return CheckCuda().is_cuda_available()


def get_cuda_info() -> List[CudaDevice]:
    return CheckCuda().get_cuda_info()

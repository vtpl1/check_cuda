import ctypes
import logging
import os
import platform
from typing import Dict, List, Union

import psutil
import pynvml as N
from cpuinfo import get_cpu_info
from singleton_decorator.decorator import singleton
import yaml

from .models import (ChannelAndNnModel, CpuInfo, CpuStatus, GpuInfo, GpuStatus, ModelCount, NnModel, NnModelMaxChannel, NnModelMaxChannelList, ProcessStatus,
                     SystemInfo, SystemStatus)

# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024


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


LOGGER = logging.getLogger(__name__)

# def get_public_ip() -> str:
#     'https://api.ipify.org?format=json'


@singleton
class GpuInfoFromNvml(object):
    def __init__(self):
        self.__is_nvml_loaded = False
        self.__gpu_processes: List[ProcessStatus] = []
        print("Starting NVML")
        try:
            N.nvmlInit()
            self.__is_nvml_loaded = True
        except Exception as e:
            print(e)

    def __del__(self):
        print("Shutting down NVML")
        if self.__is_nvml_loaded:
            N.nvmlShutdown()

    def _decode(self, b):
        if isinstance(b, bytes):
            return b.decode('utf-8')    # for python3, to unicode
        return b

    def get_gpu_status_by_gpu_id(self, index) -> Union[GpuStatus, None]:
        gpu_status = None
        if self.__is_nvml_loaded:
            gpu_status = GpuStatus(index=index)
            """Get one GPU information specified by nvml handle"""
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_status.name = self._decode(N.nvmlDeviceGetName(handle))
            gpu_status.uuid = self._decode(N.nvmlDeviceGetUUID(handle))
            try:
                gpu_status.temperature = N.nvmlDeviceGetTemperature(handle, N.NVML_TEMPERATURE_GPU)
            except N.NVMLError:
                gpu_status.temperature = None    # Not supported
            try:
                gpu_status.fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            except N.NVMLError:
                gpu_status.fan_speed = None    # Not supported
            try:
                memory = N.nvmlDeviceGetMemoryInfo(handle)   # in Bytes
                gpu_status.memory_used = memory.used // MB
                gpu_status.memory_total = memory.total // MB
            except N.NVMLError:
                gpu_status.memory_used = None    # Not supported
                gpu_status.memory_total = None

            try:
                gpu_status.utilization = N.nvmlDeviceGetUtilizationRates(handle)
            except N.NVMLError:
                gpu_status.utilization = None    # Not supported

            try:
                gpu_status.utilization_enc = N.nvmlDeviceGetEncoderUtilization(handle)
            except N.NVMLError:
                gpu_status.utilization_enc = None    # Not supported

            try:
                gpu_status.utilization_dec = N.nvmlDeviceGetDecoderUtilization(handle)
            except N.NVMLError:
                gpu_status.utilization_dec = None    # Not supported

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
                self.__gpu_processes.clear()
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
                        process = get_process_status_by_pid(nv_process.pid)
                        # Bytes to MBytes
                        # if drivers are not TTC this will be None.
                        usedmem = nv_process.usedGpuMemory // MB if \
                            nv_process.usedGpuMemory else None
                        process.gpu_memory_usage_mib = usedmem
                        process.gpu_id = index
                        self.__gpu_processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass
        return gpu_status

    def get_process_status_running_on_gpus(self) -> List[ProcessStatus]:
        return self.__gpu_processes

    def get_gpu_status(self) -> List[GpuStatus]:
        gpu_list = []
        if self.__is_nvml_loaded:
            device_count = N.nvmlDeviceGetCount()
            for index in range(device_count):
                gpu_status = self.get_gpu_status_by_gpu_id(index)
                if gpu_status:
                    gpu_list.append(gpu_status)
        return gpu_list


@singleton
class GpuInfoFromCudaLib:
    def __init__(self):
        self.__cuda = None
        self.__nvidia_device_list: List[GpuInfo] = []

    def get_gpu_info(self) -> List[GpuInfo]:
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
                        self.__nvidia_device_list.append(GpuInfo(
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
                        ))

        return self.__nvidia_device_list


def get_process_status_by_pid(pid) -> ProcessStatus:
    ps_process = psutil.Process(pid=pid)
    process = _extract_process_info(ps_process)
    return process


def get_process_status_by_name(name='python3') -> List[ProcessStatus]:
    process_list = []
    for ps_process in psutil.process_iter():
        name_, exe, cmdline = "", "", []
        try:
            name_ = ps_process.name()
            cmdline = ps_process.cmdline()
            exe = ps_process.exe()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except psutil.NoSuchProcess:
            continue
        if name == name_ or cmdline[0] == name or os.path.basename(exe) == name:
            process_list.append(_extract_process_info(ps_process))
    return process_list


def get_process_status_running_on_gpus() -> List[ProcessStatus]:
    return GpuInfoFromNvml().get_process_status_running_on_gpus()


def _extract_process_info(ps_process) -> ProcessStatus:
    process = ProcessStatus()
    try:
        process.username = ps_process.username()
    except psutil.AccessDenied:
        pass

    # cmdline returns full path;,        # as in `ps -o comm`, get short cmdnames.
    _cmdline = None
    try:
        _cmdline = ps_process.cmdline()
    except psutil.AccessDenied:
        pass

    if not _cmdline:
        # sometimes, zombie or unknown (e.g. [kworker/8:2H])
        process.command = '?'
        process.full_command = ['?']
    else:
        process.command = os.path.basename(_cmdline[0])
        process.full_command = _cmdline
    try:
        process.cpu_percent = ps_process.cpu_percent() / psutil.cpu_count()
        process.cpu_memory_usage = round((ps_process.memory_percent() / 100.0) *
                                         psutil.virtual_memory().total // MB)
    except psutil.AccessDenied:
        pass
    process.pid = ps_process.pid
    return process


def get_process_status() -> List[ProcessStatus]:
    ret = get_process_status_running_on_gpus()
    if not len(ret):
        ret = get_process_status_by_name()
    return ret


def get_cpu_status() -> CpuStatus:
    return CpuStatus(cpu_percent=psutil.cpu_percent(),
                     cpu_memory_usage_percent=psutil.virtual_memory().percent)


def get_gpu_status() -> List[GpuStatus]:
    return GpuInfoFromNvml().get_gpu_status()


def get_gpu_info() -> List[GpuInfo]:
    return GpuInfoFromCudaLib().get_gpu_info()


def get_system_status() -> SystemStatus:
    return SystemStatus(cpu=get_cpu_status(), gpus=get_gpu_status(), processes=get_process_status())


def get_cpu() -> CpuInfo:
    cpu = CpuInfo()
    try:
        cpu_info = get_cpu_info()
        cpu.name = cpu_info["brand_raw"]
        cpu.frequency = cpu_info["hz_advertised_friendly"]
        cpu.arch = cpu_info["arch"]
        cpu.bits = cpu_info["bits"]
        cpu.count = cpu_info["count"]
        cpu.vendor_id = cpu_info["vendor_id_raw"]
    except AttributeError as e:
        LOGGER.fatal(e)
    return cpu


def get_system_info() -> SystemInfo:
    return SystemInfo(host_name=platform.uname().node, os=platform.platform(), cpu=get_cpu(), gpus=get_gpu_info())


@singleton
class ChannelGpuManager:
    """
    docstring
    """
    def __init__(self) -> None:
        self.channel_to_gpu_map: Dict[ChannelAndNnModel, ModelCount] = {}
        self.gpu_id_generator = 0
        self.configuration_file_name = self.__class__.__name__ + ".yml"
        self.model_list = self.__read_default_models()
        self.number_of_gpus = len(get_gpu_info())

    def __write_default_models(self) -> NnModelMaxChannelList:
        model_list = NnModelMaxChannelList()
        model_list.models.append(NnModelMaxChannel(key=NnModel(75, 416, 416), max_channel=2))
        model_list.models.append(NnModelMaxChannel(key=NnModel(76, 416, 416), max_channel=3))

        with open(self.configuration_file_name, 'w') as outfile:
            yaml.dump(model_list.to_dict(), outfile)

        return model_list

    def __read_default_models(self) -> NnModelMaxChannelList:
        model_list = None
        try:
            with open(self.configuration_file_name, 'r') as infile:
                model_list = NnModelMaxChannelList.from_dict(yaml.safe_load(infile))
        except FileNotFoundError:
            pass
        if not model_list:
            model_list = self.__write_default_models()
        if not len(model_list.models):
            model_list = self.__write_default_models()
        return model_list
        

    def get_next_gpu_id(self) -> int:
        ret = self.gpu_id_generator
        if not self.number_of_gpus:
            self.gpu_id_generator = (self.gpu_id_generator + 1) % self.number_of_gpus
        return ret

    

def get_gpu_id_for_the_channel(channel_id: int, purpose: int, width: int, height: int, media_tpe: int = 2) -> int:
    candidate = ChannelAndNnModel(channel_id, NnModel(purpose, width, height))
    if candidate in ChannelGpuManager().channel_to_gpu_map.keys():
        x = ChannelGpuManager().channel_to_gpu_map[candidate]
        x.count = x.count + 1

    else:
        x = ModelCount(gpu_id=ChannelGpuManager().get_next_gpu_id())
        ChannelGpuManager().channel_to_gpu_map[candidate] = x
    print(candidate, x)
    return x.gpu_id

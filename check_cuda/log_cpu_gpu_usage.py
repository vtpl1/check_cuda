import logging
from threading import Event, Thread

import psutil

from .class_object_flattener import (get_flatten_keys, get_flatten_keys_list,
                                     get_flatten_values_list)
from . import controllers
from .controllers import get_system_info, get_system_status

LOGGER = logging.getLogger(__name__)
LOGGER_CPU_USAGE = logging.getLogger("cpu_usage")


class LogCpuGpuUsage(Thread):
    """
    Log CPU, GPU and memory usage
    """

    def __init__(self):
        self.__is_stop = Event()
        self.__is_already_shutting_down = False
        super().__init__()

    def run(self) -> None:
        obj = controllers.get_system_info()
        LOGGER_CPU_USAGE.info(f"============== Start ================ {obj.host_name}, {obj.os}")

        obj = controllers.get_system_status()
        header = "cpu_percent,cpu_memory_usage_percent,"
        i = 0
        for gpu in obj.gpus:
            header += f"#GPU{str(i)},index,uuid,name,utilization_gpu,utilization_enc,utilization_dec,memory_used,memory_total,"
            i += 1

        i = 0
        for process in obj.processes:
            header += f"#PROCESS{str(i)},pid,command,cpu_percent,cpu_memory_usage_mib,gpu_id,gpu_memory_usage_mib,"
            i += 1


        LOGGER_CPU_USAGE.info(header)
        while True:            
            s = f"{obj.cpu.cpu_percent},{obj.cpu.cpu_memory_usage_percent},"
            i = 0
            for gpu in obj.gpus:
                s += f"#{str(i)},{str(gpu.index)},{str(gpu.uuid)},{str(gpu.name)},{str(gpu.utilization_gpu)},{str(gpu.utilization_enc)},{str(gpu.utilization_dec)},{str(gpu.memory_used)},{str(gpu.memory_total)},"
                i += 1

            i = 0
            for process in obj.processes:
                s += f"#{str(i)},{str(process.pid)},{str(process.command)},{str(process.cpu_percent)},{str(process.cpu_memory_usage_mib)},{str(process.gpu_id)},{str(process.gpu_memory_usage_mib)},"
                i += 1            
            LOGGER_CPU_USAGE.info(s)
            if self.__is_stop.wait(1.0):
                break
            else:
                obj = controllers.get_system_status()
                continue
            

        LOGGER_CPU_USAGE.info("============== End   ================")

    def stop(self):
        if self.__is_already_shutting_down:
            return
        self.__is_already_shutting_down = True
        self.__is_stop.set()

    def __del__(self):
        self.stop()

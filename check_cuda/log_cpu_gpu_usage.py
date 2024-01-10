import logging
import time
from threading import Event, Thread

from . import controllers

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

    def stop(self):
        if self.__is_already_shutting_down:
            return
        self.__is_already_shutting_down = True
        print("Stop called")
        self.__is_stop.set()

    def __del__(self):
        self.stop()

    def run(self) -> None:
        obj_get_system_info = controllers.get_system_info()
        host_name = obj_get_system_info.host_name
        host_os = obj_get_system_info.os
        LOGGER_CPU_USAGE.info(
            f"============== Start ================ {host_name}, {host_os}"
        )

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
        db_name = "Hajmola"
        measurement_name = "cpu_gpu"
        client = None
        while True:
            s_influx = (
                f"cpu_percent={obj.cpu.cpu_percent},"
                f"cpu_memory_usage_percent={obj.cpu.cpu_memory_usage_percent}"
            )
            s = f"{obj.cpu.cpu_percent},{obj.cpu.cpu_memory_usage_percent},"
            i = 0
            for gpu in obj.gpus:
                s += (
                    f"#GPU{str(i)},"
                    f"{str(gpu.index)},"
                    f"{str(gpu.uuid)},"
                    f"{str(gpu.name)},"
                    f"{str(gpu.utilization_gpu)},"
                    f"{str(gpu.utilization_enc)},"
                    f"{str(gpu.utilization_dec)},"
                    f"{str(gpu.memory_used)},"
                    f"{str(gpu.memory_total)},"
                )
                i += 1

            i = 0
            for process in obj.processes:
                s += (
                    f"#PROCESS{str(i)},"
                    f"{str(process.pid)},"
                    f"{str(process.command)},"
                    f"{str(process.cpu_percent)},"
                    f"{str(process.cpu_memory_usage_mib)},"
                    f"{str(process.gpu_id)},"
                    f"{str(process.gpu_memory_usage_mib)},"
                )
                i += 1
            LOGGER_CPU_USAGE.info(s)
            if client:
                data = []
                t = int(time.time() * 1000)
                data_point = f"{measurement_name},host={host_name} {s_influx} {t}"
                data.append(data_point)
                # print(data_point)
                print(s)
                try:
                    client.write_points(
                        data, time_precision="ms", batch_size=1, protocol="line"
                    )
                except Exception as e:
                    print(e)
                    client = None
            if self.__is_stop.wait(1.0):
                break
            else:
                obj = controllers.get_system_status()
                continue
        if client:
            client.close()
        LOGGER_CPU_USAGE.info("============== End   ================")

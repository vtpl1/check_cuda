import logging
from threading import Event, Thread

import psutil

from .class_object_flattener import (get_flatten_keys, get_flatten_keys_list,
                                     get_flatten_values_list)
from . import controllers
from .controllers import get_system_info, get_system_status

LOGGER = logging.getLogger(__name__)
LOGGER_CPU_USAGE = logging.getLogger(__name__)


class LogCpuGpuUsage(Thread):
    """
    Log CPU, GPU and memory usage
    """

    def __init__(self):
        self.__is_stop = Event()
        self.__is_already_shutting_down = False
        super().__init__()

    def run(self) -> None:
        LOGGER_CPU_USAGE.info("============== Start ================")

        obj = controllers.get_system_info()
        d = get_flatten_keys(obj)
        LOGGER.info(get_flatten_keys_list(d))
        LOGGER.info(get_flatten_values_list(obj, d))
        obj = controllers.get_system_status()
        d = get_flatten_keys(obj)

        LOGGER_CPU_USAGE.info(get_flatten_keys_list(d))
        LOGGER_CPU_USAGE.info(get_flatten_values_list(obj, d))
        while True:
            obj = controllers.get_system_status()
            LOGGER_CPU_USAGE.info(get_flatten_values_list(obj, d))
            if self.__is_stop.wait(10.0):
                break
            else:
                continue
        LOGGER_CPU_USAGE.info("============== End   ================")

    def stop(self):
        if self.__is_already_shutting_down:
            return
        self.__is_already_shutting_down = True
        self.__is_stop.set()

    def __del__(self):
        self.stop()

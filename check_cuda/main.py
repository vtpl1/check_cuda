import logging
import logging.config
import os
import shutil
import codecs
import signal
import threading

import yaml

from .get_hardware_info import get_hardware_info
from .utils import get_session_folder
from . import log_cpu_gpu_usage

LOGGER = logging.getLogger(__name__)


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = get_session_folder() + default_path
    # value = os.getenv(env_key, None)
    # if value:
    #     path = value
    if not os.path.exists(path):
        shutil.copy('check_cuda/logging.yaml', path)

    with open(path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    # logging.basicConfig(level=default_level)

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version():
    return read("VERSION")

is_shutdown = threading.Event()


def stop_handler(*args):
    #del signal_received, frame
    LOGGER.info("")
    LOGGER.info("=============================================")
    LOGGER.info("Bradcasting global shutdown from stop_handler")
    LOGGER.info("=============================================")
    #zope.event.notify(shutdown_event.ShutdownEvent("KeyboardInterrupt received"))
    global is_shutdown
    is_shutdown.set()


def raise_unhandled_exeception_error():
    LOGGER.info("")
    LOGGER.info("=============================================")
    LOGGER.info("Bradcasting unhandled exception error")
    LOGGER.info("=============================================")
    #zope.event.notify(shutdown_event.ShutdownEvent("Unhandled global exception"))
    global is_shutdown
    is_shutdown.set()


def main():
    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    print("Using session {}".format(get_session_folder()))
    setup_logging()
    LOGGER.info("=============================================")
    LOGGER.info("              Started  {} {}               ".format(__name__, get_version()))
    LOGGER.info("=============================================")
    print("Using session {}".format(get_session_folder()))

    # try:
    #     l = log_cpu_gpu_usage.LogCpuGpuUsage()
    #     l.start()
    #     global is_shutdown
    #     while not is_shutdown.wait(10.0):
    #         continue
    #     l.stop()
    # except Exception as e:
    #     LOGGER.exception(e)
    #     # LOGGER.fatal(e)
    #     raise_unhandled_exeception_error()

    try:        
        global is_shutdown
        while not is_shutdown.wait(1.0):
            get_hardware_info()
            continue
    except Exception as e:
        LOGGER.exception(e)
        # LOGGER.fatal(e)
        raise_unhandled_exeception_error()

    
    # from cpuinfo import get_cpu_info
    # x = get_cpu_info()
    # print(x, "------------------------------------")
    # print(x['brand_raw'])
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    # print(id(CheckCuda()), CheckCuda().is_cuda_available())
    LOGGER.info("=============================================")
    LOGGER.info("              Shutdown complete {} {}               ".format(__name__, get_version()))
    LOGGER.info("=============================================")

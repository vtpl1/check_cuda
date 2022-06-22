import codecs
import logging
import logging.config
import os
import shutil
import signal
import threading

import yaml

from . import controllers, log_cpu_gpu_usage
from .utils import get_session_folder

LOGGER = logging.getLogger(__name__)


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration

    """
    path = get_session_folder() + default_path
    # value = os.getenv(env_key, None)
    # if value:
    #     path = value
    if not os.path.exists(path):
        src_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'logging.yaml'
        print("####### Copying from %s" % src_path)
        shutil.copy(src_path, path)

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

    LOGGER.info(controllers.get_system_info())
    LOGGER.info(controllers.get_system_status())
    l = None
    try:
        global is_shutdown
        l = log_cpu_gpu_usage.LogCpuGpuUsage()
        l.start()
        while not is_shutdown.wait(10.0):
            continue
    except Exception as e:
        LOGGER.exception(e)
        # LOGGER.fatal(e)
        raise_unhandled_exeception_error()
    if l is not None:
        print("Here stop")
        l.stop()

    LOGGER.info("=============================================")
    LOGGER.info("             1Shutdown complete {} {}               ".format(__name__, get_version()))
    LOGGER.info("=============================================")

# def main1():
#     signal.signal(signal.SIGINT, stop_handler)
#     signal.signal(signal.SIGTERM, stop_handler)
#     print("Using session {}".format(get_session_folder()))
#     setup_logging()
#     LOGGER.info("=============================================")
#     LOGGER.info("              Started  {} {}               ".format(__name__, get_version()))
#     LOGGER.info("=============================================")
#     print("Using session {}".format(get_session_folder()))

#     controllers.get_gpu_id_for_the_channel(0, 75, 416, 416)
#     controllers.get_gpu_id_for_the_channel(0, 75, 416, 416)

#     controllers.get_gpu_id_for_the_channel(0, 76, 416, 416)
#     controllers.get_gpu_id_for_the_channel(0, 75, 416, 416)

#     controllers.get_gpu_id_for_the_channel(0, 75, 416, 416)
#     controllers.get_gpu_id_for_the_channel(0, 75, 416, 416)
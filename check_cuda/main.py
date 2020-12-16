import logging
import logging.config
import os
import shutil

import yaml

from .get_hardware_info import get_hardware_info
from .utils import get_session_folder


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


def main():

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

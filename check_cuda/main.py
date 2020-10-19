from .get_hardware_info import get_hardware_info
import logging
import yaml
import os

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
import platform
from pathlib import Path

import PyInstaller.__main__

py_installer_seperator = ";" if platform.system().lower() == "windows" else ":"
HERE = Path(__file__).parent.absolute()
MODULE_NAME = "check_cuda"
path_to_main = f"{MODULE_NAME}/cli.py"


def install():
    PyInstaller.__main__.run(
        [
            path_to_main,
            "--collect-submodules",
            str(MODULE_NAME),
            "--add-data",
            f"{HERE}/{MODULE_NAME}/logging.yaml{py_installer_seperator}.",
            "--add-data",
            f"{HERE}/{MODULE_NAME}/VERSION{py_installer_seperator}.",
            "--onefile",
            # "--windowed",
        ]
    )

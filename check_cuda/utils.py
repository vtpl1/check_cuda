import os
import time

def get_folder(sub_folder) -> str:
    session_folder = os.path.join(os.getcwd(), sub_folder)
    if not os.path.exists(session_folder):
        try:
            os.makedirs(session_folder)
            print("{} folder created in {}".format(sub_folder, session_folder))
        except OSError as e:
            print(e)
            raise
    return session_folder + os.path.sep


def get_session_folder() -> str:
    return get_folder("session")


def get_app_folder() -> str:
    return get_folder("app")


def get_progress_folder() -> str:
    return get_session_folder()

def get_data_folder() -> str:
    return get_session_folder()

def get_config_folder() -> str:
    return get_session_folder()

def get_temp_file_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def get_final_file_name(initial_file_name: str) -> str:
    return initial_file_name + "_" + time.strftime("%Y%m%d-%H%M%S")

def get_id():
    return int(round(time.time() * 1000000))


def get_current_time():
    return int(round(time.time() * 1000))


def get_current_time_sec():
    return int(round(time.time()))

from enum import Enum, auto
import os
from pathlib import Path
from loguru import logger
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


def get_path(fpath, base_dir=os.getenv('PROJECT_DIR')):
    return os.path.join(base_dir, fpath)


def create_folder_if_not_exists(folder_path):
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=False)
        logger.info(f"Created folder {folder_path}")
    except FileExistsError:
        print(f"Folder {folder_path} already exists")


class Input(Enum):
    BOOKS = auto()
    MOVIES = auto()


def aupath(folder, file_name):
    return os.path.join(folder, file_name)

from os.path import exists
from os import mkdir
from shutil import rmtree


def clean_directories(*args: str) -> None:
    """
    Clean all directories passed
    :param args: paths to the directories
    """
    for directory in args:
        if exists(directory):
            rmtree(directory)
        mkdir(directory)

import os


def create_directories(list_dir):
    for DIR in (list_dir):
        try:
            os.mkdir(DIR)
        except(FileExistsError):
            pass
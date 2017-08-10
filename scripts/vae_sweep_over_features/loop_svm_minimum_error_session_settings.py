import os

import settings
from lib.utils.os_aux import create_directories

kfolds_folder = "kfolds"
session_name = "Sesion Sweep over svm minimum training error"

path_session = os.path.join(
    settings.path_to_general_out_folder,
    session_name)

path_kfolds_folder = os.path.join(path_session, kfolds_folder)

create_directories([path_session, path_kfolds_folder])

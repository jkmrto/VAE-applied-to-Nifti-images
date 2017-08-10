import os

import settings
from lib.utils.os_aux import create_directories

Kfolds_folder = "Kfolds index"
session_name = "Full_classification_session_with_k-folds"


path_kfolds_session_folder = os.path.join(
    settings.path_to_general_out_folder,
    session_name)

path_kfolds_folder = os.path.join(path_kfolds_session_folder, Kfolds_folder)

create_directories([path_kfolds_session_folder,path_kfolds_folder])


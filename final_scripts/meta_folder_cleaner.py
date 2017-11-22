import os
from lib.delete_pre_final_meta_data import  delete_simple_session

path_to_session = "/home/jkmrto/RemoteDisks/BiosipServer/new_folder_3/VAN-applied-to-Nifti-images/out/cvae_create_meta_nets_iter_1000-layer_22_11_2017_18:40"
session_to_clean_meta_folder = os.path.join(path_to_session, "meta")
delete_simple_session(session_to_clean_meta_folder=session_to_clean_meta_folder)
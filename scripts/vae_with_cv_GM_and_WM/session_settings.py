import os
import settings
from lib.aux_functionalities.os_aux import create_directories


CV_folder = "cv"
GM_folder = "vae_GM"
WM_folder = "vae_WM"
POST_ECONDING_FOLDER = "post_encoding"
wm_gm_cv_session_name = "Full session: GM + WM with Cross_Validation"
path_wm_gm_cv_session_folder = os.path.join(settings.path_to_general_out_folder,
                                            wm_gm_cv_session_name)

path_GM_folder = os.path.join(path_wm_gm_cv_session_folder, GM_folder)
path_WM_folder = os.path.join(path_wm_gm_cv_session_folder, WM_folder)
path_cv_folder = os.path.join(path_wm_gm_cv_session_folder, CV_folder)
path_post_encoding_folder = os.path.join(path_wm_gm_cv_session_folder,
                                         POST_ECONDING_FOLDER)


create_directories([path_wm_gm_cv_session_folder, path_GM_folder,
                    path_WM_folder,
                    path_cv_folder, path_post_encoding_folder])
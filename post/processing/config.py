import sys,os
sys.path.append('backend\djangoreactapi')
from djangoreactapi import settings

# PATH #

# Django url
LOCALHOST = settings.LOCALHOST

# media local path / url
MEDIA_DIR = settings.MEDIA_ROOT
MEDIA_URL = settings.MEDIA_URL

# media/images path / url
USER_IMAGE_DIR = os.path.join(MEDIA_DIR,"images") 
USER_IMAGE_URL = os.path.join(MEDIA_URL,"images") 

# result path / url
OUT_URL = os.path.join(MEDIA_URL,"out") 
OUT_DIR = os.path.join(MEDIA_DIR,"out")
os.makedirs(OUT_DIR,exist_ok=True)

# /backend
BACKEND_DIR = settings.BASE_DIR

# Network #
IMAGE_SHAPE = (512,512,3)
MODE_SAVE_DCT = "post/processing/pre_trained/result/logs/defactor/DCT_RRUnet"
MODE_SAVE_RRU = "post/processing/pre_trained/result/logs/defactor/Ringed_Res_Unet"
MODE_SAVE_U2 = "post/processing/pre_trained/result/logs/defactor/U2" # <- 미정 
MODE_NAME_DCT = "DCT-[val_dice]-0.9011-[train_loss]-0.0902-ep6.pkl"
MODE_NAME_RRU = "defactor-[val_dice]-0.7420-[train_loss]-0.0753.pkl"
MODE_NAME_U2 = ""

# "http://127.0.0.1:8000/media/images/a2_APHuk9o.jpg"
def process_raw_path(raw_path,out = False):
    """  From React path
        "C/:fakeepath/'imageName.jpg' => available url

    Args:
        raw_path (str): frontend to backend / backend to frontend path
        out (bool, optional): True : backend to frontend , False : frontend to backend
        Defaults to False.

    Returns:
        str: url or path, file_name
    """
    file_name = str(raw_path).split('\\')[-1]    
    
    if out: # store path
        url = os.path.join(LOCALHOST+
                        OUT_URL,
                        file_name)
    else: # react src url
        url = os.path.join(USER_IMAGE_DIR,
                        file_name)
    
    return url,file_name
    

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # matplotlib on thread
from . import config as cfg
import PIL.Image as Image
import torch
import matplotlib.pyplot as plt
import numpy as np

def draw_heatmap(pred,img,img_name):
    plt.switch_backend('AGG')
    f = plt.figure(figsize=(12,10),dpi=96)
    plt.imshow(img)
    pred[0,0,0] = 0.99 # colorbar low maximum 베제
    plt.imshow(pred ,alpha=0.6,cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.close()
    f.canvas.draw()
    ax = f.canvas.renderer._renderer
    heatmap = Image.fromarray(np.array(ax)[100:-100,100:-100,:3])
    store_path = os.path.join(cfg.OUT_DIR,f'{img_name}_result.jpg') 
    heatmap.convert('RGB').save(store_path)	
    heatmap_path,_ = cfg.process_raw_path(store_path,out=True) 

    return heatmap_path

    # 위조 영역 pixel 값 평균
def mean_proba(pred):
    H,W,C = np.where((pred>=0.5))
    if np.size(H):
        return np.mean(pred[H,W,C])
    else:
        return 0

def get_result(pred,img,img_name):
    # isForgery
    print('get result')
    isForgery = True in (pred>0.5)
    proba = mean_proba(pred)
    heatmap = draw_heatmap(pred,img,img_name)
    
    return heatmap, str(proba*100.0)[:5], str(isForgery)



    
def test_process(raw_path):
    # load user data x         
    _,img_name = cfg.process_raw_path(raw_path,out=False) # user image path
    img_name = img_name.split('.')[0]
    user_img = Image.open(os.path.join(cfg.USER_IMAGE_DIR,img_name))
    
    # load pre-trained model 
    ## model = torch.load(mm.pth)
    
    # model predict x -> y : 
    ## pred = model(user_img)
    ## heatmap , proba , isForgery = pred 
    
    # store images/out/y
    # result = PIL(pred) 
    ## result.save(cfg.OUT_DIR+'/{img_name}_result.jpg')
    
    heatmap_test = os.path.join(cfg.OUT_DIR,f'{img_name}_result.jpg')
    proba_test = 0.8
    isForgery_test = True
    
    return heatmap_test, proba_test, isForgery_test
# for deepLab
def seg_result(pred,input_image_size,show=False):
    # create a color pallette, selecting a color for each class
    print('seg_result')
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(pred.byte().cpu().numpy()).resize(input_image_size)
    r.putpalette(colors)
    
    if show:
        plt.imshow(r)
        plt.show()
    return r


import os
from . import config as cfg
import PIL.Image as image
from . import network as net
from . import utill as u

class Main():
    def __init__(self):
        self.image_dir = cfg.USER_IMAGE_DIR
        self.result_dir = cfg.OUT_DIR
        
    def get(self):
        print('GET')

    def run(self,raw_path): # raw_path: react에서 넘어온 secure path
        
        # 0. 경로 수정
        user_path,img_name = cfg.process_raw_path(raw_path,out=False)
        img_name = img_name.split('.')[0]

        print("llllllllllllll",user_path)

        # 1. Network Procees 
        # 1-1. load model                             
        model = net.get_model()             
        # 1-2. pred : p.shape (512,512,1)                                
        pred,resized_img = net.pred_for_dct_rrunet(model,user_path)  
        
        # 2. Get Result [heatmap_image, proba, isForgery] 
        heatmap_path, proba, isForgery = u.get_result(pred,resized_img,img_name)
        
        print(heatmap_path, proba, isForgery)
        # Return 
        return heatmap_path, proba, isForgery

    def example_run(self,raw_path):
        # user image path 
        user_path,img_name = cfg.process_raw_path(raw_path,out=False)
        img_name = img_name.split('.')[0]

        # load model
        model = net.get_model_example()
    
        # pred
        pred, img_size = net.pred(model,user_path)
        
        # result [seg_image, proba, isForgery] 예시 이므로 proba,isForgery 생략
        seg_image = u.seg_result(pred,img_size,show=False)
        # seg_image, proba, isForgery = net.seg_result(pred,img_size,show=True)
        
        # store out image(segmentation(heatmap))
        print('save')
        store_path = os.path.join(cfg.OUT_DIR,f'{img_name}_result.jpg') 
        seg_image.convert('RGB').save(store_path)
        
        # return 
        seg_test,_ = cfg.process_raw_path(store_path,out=True) 
        proba_test = 0.8
        isForgery_test = True
        
        return seg_test, proba_test, isForgery_test
        
if __name__ == '__main__':
    sess = Main()
    h,p,b = sess.run("http://127.0.0.1:8000/media/images/raccoon.jpg")
    print(f"""
          heamap : {h}
          proba  : {p}
          isF    : {b} 
          """)
    
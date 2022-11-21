# -*- coding: utf-8 -*-
# network define file
"""
네트워크 load 및 predict 진행
"""
import os
from . import config as cfg
from .pre_trained.model.unet_model import Ringed_Res_Unet,DCT_RRUnet # pre-trained folder 바꾸기(network.py에서만 폴더 접근)
import numpy as np
import jpegio
import torch
from PIL import Image
import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt
import random
 
def get_model(train=False,gpu = False,model_name='DCT_RRUnet'):
    # model load(model)
    print('get_model')
  
    if 'DCT_RRUnet' == model_name:
        model = DCT_RRUnet()
        model.load_state_dict(torch.load(
            f'{cfg.BACKEND_DIR}/{cfg.MODE_SAVE_DCT}/{cfg.MODE_NAME_DCT}',
        map_location=torch.device('cpu')))
    elif 'RRUnet' == model_name:
        model = DCT_RRUnet()
        model.load_state_dict(torch.load(
            f'{cfg.BACKEND_DIR}/{cfg.MODE_SAVE_RRU}/{cfg.MODE_NAME_RRU}',
        map_location=torch.device('cpu')))
    elif 'U2'== model_name:
        model = None # unet2 plus
        model.load_state_dict(torch.load(
            f'{cfg.BACKEND_DIR}/{cfg.MODE_SAVE_U2}/{cfg.MODE_NAME_U2}',
        map_location=torch.device('cpu')))
    
    if train:
        model.train()
    else:
        model.eval()
    if gpu:
        if torch.cuda.is_available():
            model.cuda()
            print('model device : cuda')
            
    return model

def jpg_or_not(filename):
    """
    if f = jpg :return f
    else f to jpg
    """
    print('pred')
    f = filename
    if not filename.endswith('jpg'):
        outfile_path = filename[:-3] + "jpg"
        input_image = cv.imread(filename)
        input_image = cv.cvtColor(input_image,cv.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        input_image.save(outfile_path, "JPEG", quality=100)
        f = outfile_path
    
    return f

def pred(model,filename:str):
    print('pred')
    f = jpg_or_not(filename)
        
    input_image = Image.open(f)
    input_image = input_image.convert("RGB")
    
    preprocess = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SHAPE[0],cfg.IMAGE_SHAPE[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    output = torch.sigmoid(output)
    output = output.squeeze(0)
    output_predictions = output.permute(1,2,0).detach().numpy()
    pre_image = input_tensor.permute(1,2,0).numpy()
    
    print(output_predictions.shape)
    return output_predictions,pre_image


def pred_for_dct_rrunet(model,filename):
    print('pred')
    f = jpg_or_not(filename)
    jpg_artifact,_,qtable = create_tensor(f,None)
    
    jpg_artifact = jpg_artifact.unsqueeze(0)
    qtable = qtable.unsqueeze(0)
    
    if torch.cuda.is_available():
        jpg_artifact = jpg_artifact.to('cuda')
        qtable = qtable.to('cuda')
        model.cuda()
    
    with torch.no_grad():
        output = model(jpg_artifact,qtable)
    output = torch.sigmoid(output)
    output = output.squeeze(0)
    output_predictions = output.permute(1,2,0).detach().cpu().numpy()
    pre_image = jpg_artifact.cpu().squeeze(0)[:3,:,:].permute(1,2,0).numpy()
    
    print(output_predictions.shape)
    return output_predictions,pre_image

def preprocess_input(f):
    jpg_artifact,_,qtable = create_tensor(f,None)
    return jpg_artifact,qtable

# for DCT_RRUnet
def get_jpeg_info(im_path:str):
        """
        :param im_path: JPEG image path
        :return: DCT_coef (Y,Cb,Cr), qtables (Y,Cb,Cr)
        """
        num_channels =  1#DCT_channels
        
        print(im_path)
        
        jpeg = jpegio.read(im_path)

        # determine which axes to up-sample
        ci = jpeg.comp_info
        need_scale = [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
        if num_channels == 3:
            if ci[0].v_samp_factor == ci[1].v_samp_factor == ci[2].v_samp_factor:
                need_scale[0][0] = need_scale[1][0] = need_scale[2][0] = 2
            if ci[0].h_samp_factor == ci[1].h_samp_factor == ci[2].h_samp_factor:
                need_scale[0][1] = need_scale[1][1] = need_scale[2][1] = 2
        else:
            need_scale[0][0] = 2
            need_scale[0][1] = 2

        # up-sample DCT coefficients to match image size
        DCT_coef = []
        for i in range(num_channels):
            r, c = jpeg.coef_arrays[i].shape
            coef_view = jpeg.coef_arrays[i].reshape(r//8, 8, c//8, 8).transpose(0, 2, 1, 3)
            # case 1: row scale (O) and col scale (O)
            if need_scale[i][0]==1 and need_scale[i][1]==1:
                out_arr = np.zeros((r * 2, c * 2))
                out_view = out_arr.reshape(r * 2 // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, ::2, :, :] = coef_view[:, :, :, :]
                out_view[::2, 1::2, :, :] = coef_view[:, :, :, :]
                out_view[1::2, 1::2, :, :] = coef_view[:, :, :, :]

            # case 2: row scale (O) and col scale (X)
            elif need_scale[i][0]==1 and need_scale[i][1]==2:
                out_arr = np.zeros((r * 2, c))
                DCT_coef.append(out_arr)
                out_view = out_arr.reshape(r*2//8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[::2, :, :, :] = coef_view[:, :, :, :]
                out_view[1::2, :, :, :] = coef_view[:, :, :, :]

            # case 3: row scale (X) and col scale (O)
            elif need_scale[i][0]==2 and need_scale[i][1]==1:
                out_arr = np.zeros((r, c * 2))
                out_view = out_arr.reshape(r // 8, 8, c * 2 // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, ::2, :, :] = coef_view[:, :, :, :]
                out_view[:, 1::2, :, :] = coef_view[:, :, :, :]

            # case 4: row scale (X) and col scale (X)
            elif need_scale[i][0]==2 and need_scale[i][1]==2:
                out_arr = np.zeros((r, c))
                out_view = out_arr.reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
                out_view[:, :, :, :] = coef_view[:, :, :, :]

            else:
                raise KeyError("Something wrong here.")

            DCT_coef.append(out_arr)

        # quantization tables
        qtables = [jpeg.quant_tables[ci[i].quant_tbl_no].astype(np.float) for i in range(num_channels)]

        return DCT_coef, qtables
    
def create_tensor(im_path, mask, _crop_size=(512,512), _grid_crop=True, _blocks=('RGB','DCTvol', 'qtable'), DCT_channels=1):
    ignore_index = 0

    img_RGB = np.array(Image.open(im_path).convert("RGB"))

    h, w = img_RGB.shape[0], img_RGB.shape[1]

    if 'DCTcoef' in _blocks or 'DCTvol' in _blocks or 'rawRGB' in _blocks or 'qtable' in _blocks:
        DCT_coef, qtables = get_jpeg_info(im_path)

    if mask is None:
        mask = np.zeros((h, w))

    if _crop_size is None and _grid_crop:
        crop_size = (-(-h//8) * 8, -(-w//8) * 8)  # smallest 8x8 grid crop that contains image
    elif _crop_size is None and not _grid_crop:
        crop_size = None  # use entire image! no crop, no pad, no DCTcoef or rawRGB
    else:
        crop_size = _crop_size

    if crop_size is not None:
        # Pad if crop_size is larger than image size
        if h < crop_size[0] or w < crop_size[1]:
            # pad img_RGB
            temp = np.full((max(h, crop_size[0]), max(w, crop_size[1]), 3), 127.5)
            temp[:img_RGB.shape[0], :img_RGB.shape[1], :] = img_RGB
            img_RGB = temp

            # pad mask
            temp = np.full((max(h, crop_size[0]), max(w, crop_size[1])), ignore_index)  # pad with ignore_index(-1)
            temp[:mask.shape[0], :mask.shape[1]] = mask
            mask = temp

            # pad DCT_coef
            if 'DCTcoef' in _blocks or 'DCTvol' in _blocks or 'rawRGB' in _blocks:
                max_h = max(crop_size[0], max([DCT_coef[c].shape[0] for c in range(DCT_channels)]))
                max_w = max(crop_size[1], max([DCT_coef[c].shape[1] for c in range(DCT_channels)]))
                for i in range(DCT_channels):
                    temp = np.full((max_h, max_w), 0.0)  # pad with 0
                    temp[:DCT_coef[i].shape[0], :DCT_coef[i].shape[1]] = DCT_coef[i][:, :]
                    DCT_coef[i] = temp

        # Determine where to crop
        if _grid_crop:
            s_r = (random.randint(0, max(h - crop_size[0], 0)) // 8) * 8
            s_c = (random.randint(0, max(w - crop_size[1], 0)) // 8) * 8
        else:
            s_r = random.randint(0, max(h - crop_size[0], 0))
            s_c = random.randint(0, max(w - crop_size[1], 0))

        # crop img_RGB
        img_RGB = img_RGB[s_r:s_r+crop_size[0], s_c:s_c+crop_size[1], :]

        # crop mask
        mask = mask[s_r:s_r + crop_size[0], s_c:s_c + crop_size[1]]

        # crop DCT_coef
        if 'DCTcoef' in _blocks or 'DCTvol' in _blocks or 'rawRGB' in _blocks:
            for i in range(DCT_channels):
                DCT_coef[i] = DCT_coef[i][s_r:s_r+crop_size[0], s_c:s_c+crop_size[1]]
            t_DCT_coef = torch.tensor(DCT_coef, dtype=torch.float)  # final (but used below)

    # handle 'RGB'
    if 'RGB' in _blocks:
        t_RGB = (torch.tensor(img_RGB.transpose(2,0,1), dtype=torch.float)-127.5)/127.5  # final

    # handle 'DCTvol'
    if 'DCTvol' in _blocks:
        T = 20
        t_DCT_vol = torch.zeros(size=(T+1, t_DCT_coef.shape[1], t_DCT_coef.shape[2]))
        t_DCT_vol[0] += (t_DCT_coef == 0).float().squeeze()
        for i in range(1, T):
            t_DCT_vol[i] += (t_DCT_coef == i).float().squeeze()
            t_DCT_vol[i] += (t_DCT_coef == -i).float().squeeze()
        t_DCT_vol[T] += (t_DCT_coef >= T).float().squeeze()
        t_DCT_vol[T] += (t_DCT_coef <= -T).float().squeeze()

    # create tensor
    img_block = []
    for i in range(len(_blocks)):
        if _blocks[i] == 'RGB':
            img_block.append(t_RGB)
        elif _blocks[i] == 'DCTcoef':
            img_block.append(t_DCT_coef)
        elif _blocks[i] == 'DCTvol':
            img_block.append(t_DCT_vol)
        elif _blocks[i] == 'qtable':
            continue
        else:
            raise KeyError("We cannot reach here. Something is wrong.")

    # final tensor
    tensor = torch.cat(img_block)

    if 'qtable' not in _blocks:
        return tensor, torch.tensor(mask, dtype=torch.float32), 0
    else:
        return tensor, torch.tensor(mask, dtype=torch.float32), torch.tensor(qtables[:DCT_channels], dtype=torch.float)



# Example ______________________________________________________________________________
from torchvision.models.segmentation import deeplabv3_resnet101    

def get_model_example():
    print('get_model')
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model



def pred_test(model,filename):
    print('pred')
    input_image = Image.open(filename)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions,input_image.size


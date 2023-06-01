# -*- coding: utf-8 -*-  

"""
Created on 2021/4/14

@author: Ruoyu Chen
"""

import numpy as np
import argparse
import os
import torch
import random
import cv2
import pickle

import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import auc
from prettytable import PrettyTable

from iresnet import iresnet18,iresnet34,iresnet50,iresnet100,iresnet200
from resnet import resnet50
from align import Face_Align

import sys

shape_predictor_path = "./face-alignment-dlib/shape_predictor_68_face_landmarks.dat"

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

class VGGFace2_verifacation(object):
    def __init__(self,vggnet):
        self.net = vggnet
        self.net.eval()
        self.hook_feature = None
    def _register_hook(self,net,layer_name):
        for (name, module) in net.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._get_features_hook)
    def _get_features_hook(self,module, input, output):
        self.hook_feature = output.view(output.size(0), -1)[0]
    def __call__(self,input):
        self._register_hook(self.net,"avgpool")
        self.net.zero_grad()
        self.net(input)
        return torch.unsqueeze(self.hook_feature, dim=0)

def read_datasets(real_dir,fake_dir):
    real_dir_lists = os.listdir(real_dir)
    fake_dir_lists = os.listdir(fake_dir)

    images_dir = []
    # real
    for real in real_dir_lists:
        selected = [x for x in fake_dir_lists if x.split("_")[0]==real.split("-")[0]]
        for fake in selected:
            images_dir.append([os.path.join(real_dir,real),os.path.join(fake_dir,fake),1])
    # fake
    length = len(images_dir)
    i = 0
    while(True):
        data1 = random.choice(real_dir_lists)
        data2 = random.choice(real_dir_lists)
        
        images_dir.append([os.path.join(real_dir,data1),os.path.join(real_dir,data2),0])
        i+=1
        if i == 2*length:
            break
    return images_dir

def get_net(net_type):
    arcface_r50_path = "./pre-trained/ms1mv3_arcface_r50_fp16/backbone.pth"
    arcface_r100_path = "./pre-trained/ms1mv3_arcface_r100_fp16/backbone.pth"
    cosface_r50_path = "./pre-trained/glint360k_cosface_r50_fp16_0.1/backbone.pth"
    cosface_r100_path = "./pre-trained/glint360k_cosface_r100_fp16_0.1/backbone.pth"
    if net_type == "ArcFace-r50":
        net = iresnet50()
        net.load_state_dict(torch.load(arcface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()    
    elif net_type == "ArcFace-r100":
        net = iresnet100()
        net.load_state_dict(torch.load(arcface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "CosFace-r50":
        net = iresnet50()
        net.load_state_dict(torch.load(cosface_r50_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "CosFace-r100":
        net = iresnet100()
        net.load_state_dict(torch.load(cosface_r100_path))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()
    elif net_type == "VGGFace2":
        from resnet import resnet50
        weight_path = "./pre-trained/resnet50_scratch_weight.pkl"
        vggnet = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        vggnet.load_state_dict(weights, strict=True)

        if torch.cuda.is_available():
            vggnet.cuda()
        net = VGGFace2_verifacation(vggnet)
    return net

def Image_Preprocessing(net_type,image):
    '''
    Precessing the input images
        image_dir: single image input path, such as "/home/xxx/10.jpg"
    '''
    if net_type in ['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100"]:
        image = Image.fromarray(cv2.cvtColor(np.uint8(image),cv2.COLOR_BGR2RGB))
        return transforms(image.resize((112, 112), Image.BILINEAR))
    elif net_type in ["VGGFace2","VGGFace2-verification"]:
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        assert image is not None
        image = cv2.resize(image,(224,224))
        image = image.astype(np.float32)
        image -= mean_bgr
        # H * W * C   -->   C * H * W
        image = image.transpose(2,0,1)
        return torch.tensor(image)

def verification(args,images_dir):
    net = get_net(args.Net_type)

    similarity = []
    label = []

    align = Face_Align(shape_predictor_path)

    for data in tqdm(images_dir):
        path1 = data[0]
        path2 = data[1]
        # image1 = Image.open(path1).resize((112, 112), Image.BILINEAR)
        # image2 = Image.open(path2).resize((112, 112), Image.BILINEAR)
        # image1 = transforms(image1)
        # image2 = transforms(image2)
        # image1 = align(cv2.imread(path1))
        # image2 = align(cv2.imread(path2))
        try:
            if data[2] == 1:
                image1,image2 = align.real_fake(cv2.imread(path1),cv2.imread(path2))
            else:
                image1 = align(cv2.imread(path1))
                image2 = align(cv2.imread(path2))
        
            image1 = Image_Preprocessing(args.Net_type,image1)
            image2 = Image_Preprocessing(args.Net_type,image2)

            output1 = F.normalize(net(torch.unsqueeze(image1, dim=0).cuda()),p=2,dim=1)
            output2 = F.normalize(net(torch.unsqueeze(image2, dim=0).cuda()),p=2,dim=1)

            similar = torch.cosine_similarity(output1[0], output2[0], dim=0).item()

            similarity.append(similar)
            label.append(data[2])
        except:
            pass
    return similarity,label

def AUC(score,label):
    score = np.array(score)
    label = np.array(label)

    x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1,0.2,0.4,0.6,0.8,1]
    tpr_fpr_table = PrettyTable(map(str, x_labels))
    
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr

    tpr_fpr_row = []
    
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row(tpr_fpr_row)

    print(tpr_fpr_table)
    print("ROC AUC: {}".format(roc_auc))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Verification List')
    # general
    parser.add_argument('--Net-type',
                        type=str,
                        default='VGGFace2',
                        choices=['ArcFace-r50','ArcFace-r100',"CosFace-r50","CosFace-r100","VGGFace2"],
                        help='Which network using for face verification.')
    parser.add_argument('--real-dir',
                        type=str,
                        default="./stargan_vgg/results_all_1/real_ima",
                        help='')
    parser.add_argument('--fake-dir',
                        type=str,
                        default="./stargan_vgg/results_all_1/fake_ima",
                        help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    images_dir = read_datasets(real_dir = args.real_dir,
                               fake_dir = args.fake_dir)
    similarity,label = verification(args,images_dir)


    AUC(similarity,label)

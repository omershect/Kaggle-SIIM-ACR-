import sys
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
from os.path import isfile

import scipy as sp
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import cv2 

batch_size = 16 

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True

test = '../input/aptos2019-blindness-detection/test_images/'

############################################################
# Pytorch Infernce Kernel For Kaggle APTOS 2019            #
# This code uses various models which are Ensembled         #
# Also Using TTA                                           #
############################################################




### Data Set Loader 
class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return {'image': image}
    

package_path = '../input/efficientpytorch/'
sys.path.append(package_path)
from efficientnetpytorch import EfficientNet


######### EFF 256 Models ########################256#################################
    
#### Load Model 0  ############################    
model0 = EfficientNet.from_name('efficientnet-b5')
in_features = model0._fc.in_features
model0._fc = nn.Linear(in_features, 1)

model0.load_state_dict(torch.load("../input/model-res-256/efficientNet5.fold2.best.pt"))
model0 = model0.to(device)

print("load model 0 ")



#### Load Model 1 ############################    
model1 = EfficientNet.from_name('efficientnet-b4')
in_features = model1._fc.in_features
model1._fc = nn.Linear(in_features, 1)

model1.load_state_dict(torch.load("../input/model-res-256/effb4oldtrearlystop_weight_bestf02.pt"))
model1 = model1.to(device)

print("load model 1 ")




#### Load Model 2 ############################    
model2 = EfficientNet.from_name('efficientnet-b4')
in_features = model2._fc.in_features
model2._fc = nn.Linear(in_features, 1)

model2.load_state_dict(torch.load("../input/model-res-256/weight_best_fold0_crop_center14.pt"))
model2 = model2.to(device)

print("load model 2 ")


######### EFF B3 300 Models #########################################################

#### Load Model 3 ############################    
model3 = EfficientNet.from_name('efficientnet-b3')
in_features = model3._fc.in_features
model3._fc = nn.Linear(in_features, 1)

model3.load_state_dict(torch.load("../input/effb3-300-models/efficientNet3-res-300.fold2.best.pt"))
model3 = model3.to(device)

print("load model 3 ")



#### Load Model 4 ############################    
model4 = EfficientNet.from_name('efficientnet-b3')
in_features = model4._fc.in_features
model4._fc = nn.Linear(in_features, 1)

model4.load_state_dict(torch.load("../input/effb3-300-models/efficientNet3-res-300.fold0.best.pt"))
model4 = model4.to(device)

print("load model 4 ")



#### Load Model 5 ############################    
model5 = EfficientNet.from_name('efficientnet-b3')
in_features = model5._fc.in_features
model5._fc = nn.Linear(in_features, 1)

model5.load_state_dict(torch.load("../input/effb3-300-models/efficientNet3-res-300.fold3.best.pt"))
model5 = model5.to(device)

print("load model 5 ")



######### EFF B4 380 Models #########################################################


#### Load Model 6 ############################    
model6 = EfficientNet.from_name('efficientnet-b4')
in_features = model6._fc.in_features
model6._fc = nn.Linear(in_features, 1)

model6.load_state_dict(torch.load("../input/effb4-380-models/effb4_380_weight_best8_v2.pt"))
model6 = model6.to(device)

print("load model 6 ")



#### Load Model 7 ############################    
model7 = EfficientNet.from_name('efficientnet-b4')
in_features = model7._fc.in_features
model7._fc = nn.Linear(in_features, 1)

model7.load_state_dict(torch.load("../input/effb4-380-models/effb4_380_fold2_weight_best7_v1.pt"))
model7 = model7.to(device)

print("load model 7 ")






################## B5  456 #####################################################

#### Load Model 8 ############################    
model8 = EfficientNet.from_name('efficientnet-b5')
in_features = model8._fc.in_features
model8._fc = nn.Linear(in_features, 1)

model8.load_state_dict(torch.load("../input/effb5-456-models/EFFB5_456_fold2_weight_best14_03_09.pt"))
model8 = model8.to(device)

print("load model8 ")



################## B6  512 #####################################################

#### Load Model 9 ############################    
model9 = EfficientNet.from_name('efficientnet-b6')
in_features = model9._fc.in_features
model9._fc = nn.Linear(in_features, 1)

model9.load_state_dict(torch.load("../input/effb6-res-512/weight_best12.pt"))
model9 = model9.to(device)

print("load model9 ")


### Image Pre Processing 


def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img



#Prepare Model for Prediction  

    
for i in range(10):
    
    (vars()["model"+str(i)]).eval()
    for param in (vars()["model"+str(i)]).parameters():
        param.requires_grad = False
    print("complete eval model:",i)
        



def expand_path(p):
    p = str(p)
    if isfile(test + p + ".png"):
        return test + (p + ".png")
    return p

class MyDataset(Dataset):
    
    def __init__(self, dataframe,image_size, transform=None):
        self.df = dataframe
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.diagnosis.values[idx]
        label = np.expand_dims(label, -1)
        
        p = self.df.id_code.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'), image_size = 256,transform=test_transform)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#Predict
 
for j in range(3):
    
   
    
    vars()["test_preds"+str(j)] = np.zeros((len(testset), 1))
    print("predict :",j)
    for i, data in enumerate(test_loader):
        
        images, _ = data
        images = images.cuda()
        if j == 0 :
            pred = model0(images)
        if j == 1  : 
            pred = model1(images)
        if j == 2  : 
            pred = model2(images)
      
       
       
       
        
        vars()["test_preds"+str(j)][i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
 
                                  
testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'),image_size = 300,transform=test_transform)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

for j in range(3,6):
    
   
    
    vars()["test_preds"+str(j)] = np.zeros((len(testset), 1))
    print("predict :",j)
    for i, data in enumerate(test_loader):
        
        images, _ = data
        images = images.cuda()
        if j == 3 :
            pred = model3(images)
        if j == 4 :
            pred = model4(images)
       
        if j == 5 :
            pred = model5(images)
       
       
       
       
        
        vars()["test_preds"+str(j)][i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
        
 
                                  
                                  
 
testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'),image_size = 380, transform=test_transform)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

for j in range(6,8):
    
   
    
    vars()["test_preds"+str(j)] = np.zeros((len(testset), 1))
    print("predict :",j)
    for i, data in enumerate(test_loader):
        
        images, _ = data
        images = images.cuda()
        if j == 6 :
            pred = model6(images)
        if j == 7  : 
            pred = model7(images)
        
       
       
        
        vars()["test_preds"+str(j)][i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
        
 

testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'),transform=test_transform,image_size = 456)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

for j in range(8,9):
    
   
    
    vars()["test_preds"+str(j)] = np.zeros((len(testset), 1))
    print("predict :",j)
    for i, data in enumerate(test_loader):
        
        images, _ = data
        images = images.cuda()
        if j == 8 :
            pred = model8(images)
            
       
       
       
        
        vars()["test_preds"+str(j)][i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
        
 
testset        = MyDataset(pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv'),transform=test_transform,image_size = 512)
test_loader    = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

for j in range(9,10):
    
   
    
    vars()["test_preds"+str(j)] = np.zeros((len(testset), 1))
    print("predict :",j)
    for i, data in enumerate(test_loader):
        
        images, _ = data
        images = images.cuda()
        if j == 9 :
            pred = model9(images)
            
       
       
       
        
        vars()["test_preds"+str(j)][i * batch_size:(i + 1) * batch_size] = pred.detach().cpu().squeeze().numpy().reshape(-1, 1)
      


#Ensemble
test_pred_mean = (test_preds0 + test_preds1 + test_preds2+ test_preds3+ test_preds4 + test_preds5 + test_preds6 +test_preds7 +test_preds8 + test_preds9 ) / 10.0

coef = [0.57, 1.37, 2.57, 3.57]

def convertToClasses(floatInput, coef = [0.57, 1.37, 2.57, 3.57]):
    num = floatInput.shape[0]
    output = np.asarray([4]*num, dtype=np.int64)
    
    for i, pred in enumerate(floatInput):
        if pred < coef[0]:
            output[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            output[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            output[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            output[i] = 3
    return output


submission = pd.DataFrame({'id_code':pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv').id_code.values,
                          'diagnosis':convertToClasses(test_pred_mean)})

#print(submission.head())
submission.to_csv('submission.csv', index=False)

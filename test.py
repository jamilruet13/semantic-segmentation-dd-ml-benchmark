# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:01:03 2020

@author: Tanya Joon
"""

import tensorflow as tf
from tensorflow.keras import models
from libs import inference_keras
#import wandb
from libs import scoring
import Main
import cv2
import numpy as np


from libs.config import predict_ids

def IOU(predsfile,mask):
    prediction = cv2.imread(predsfile)
    #print(prediction)
    print(prediction.shape)
    target = cv2.imread(mask)
    #print(target)
    print(target.shape)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def largest_of_three(num1,num2,num3):
    if (num1 >= num2) and (num1 >= num3):
        largest = num1
        key = 'res_score'
    elif (num2 >= num1) and (num2 >= num3):
        largest = num2
        key = 'vgg_score'
    else:
        largest = num3
        key = 'gmm_score'
    return largest, key

model_res = models.load_model('model-resnet18-10e.h5', custom_objects={'tf': tf}, compile= False)
model_res.summary()
model_vgg = models.load_model('model-vgg16-10e.h5', custom_objects={'tf': tf}, compile= False)
model_vgg.summary()

dataset = 'dataset-medium'
config = {
        'name' : 'baseline-keras',
        'dataset' : dataset,
    }

#wandb.init(config=config)

inference_keras.run_inference(dataset, model=model_res, basedir='predictions_resnet18')
score, _ = scoring.score_predictions(dataset, basedir='predictions_resnet18')
print(score)
inference_keras.run_inference(dataset, model=model_vgg, basedir='predictions_vgg16')
score, _ = scoring.score_predictions(dataset, basedir='predictions_vgg16')
print(score)

acc_res=[]
acc_gmm=[]
acc_vgg=[]
acc=[]
for predict in predict_ids: #or some list
    img_res = f'./predictions_resnet18/{predict}-prediction.png'
    img_vgg = f'./predictions_vgg16/{predict}-prediction.png'
    img_gmm = Main.runcluster(dataset,predict,basedir='predictions_cluster')
    img_gmm = f'./predictions_cluster/{predict}-prediction.png'
    img_lab = f'{dataset}/labels/{predict}-label.png'

    #print(img_lab)
    res_score = IOU(img_res,img_lab)
    print(res_score)
    vgg_score = IOU(img_vgg,img_lab)
    print(vgg_score)
    #gmm_score=1
    gmm_score = IOU(img_gmm,img_lab)
    print(gmm_score)

    large, key = largest_of_three(res_score,vgg_score,gmm_score)

    print(key)
    print(large)
    acc_res.append(res_score)   
    acc_vgg.append(vgg_score)
    acc_gmm.append(gmm_score)   
    acc.append(large)
    print(predict)

totalavg_res=np.mean(acc_res)
totalavg_vgg=np.mean(acc_vgg)
totalavg_gmm=np.mean(acc_gmm)
totalavg=np.mean(acc)
print('Resnet accuracy')
print(totalavg_res)
print('VGG accuracy')
print(totalavg_vgg)
print('GMM accuracy')
print(totalavg_gmm)
print('Total accuracy')
print(totalavg)



    

    
    











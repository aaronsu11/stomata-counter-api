# -*- coding: utf-8 -*-
"""
Author: Hiranya Jayakody. Sept 2020.
function created for hosting on AWS lambda services
"""

import os
import cv2
import glob
import sys
import json
import numpy as np
import imutils
import statistics as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib, utils
from matplotlib import pyplot as plt


def lambda_handler(path):
    """
    The main entry point to the lambda function

    """
    CWD = path #TODO:path to working folder on s3 bucket
    STOMATA_WEIGHTS_PATH = os.path.join(CWD,'weights/2020_mask_rcnn_stomata_51.h5') #TODO: path to model h5 file on s3 bucket
    DATASET_DIR = os.path.join(CWD,'images/') #TODO: modify to match s3 bucket
    INPUT_IMG_DIR = os.path.join(DATASET_DIR,'test/') #TODO: modify to match s3 bucket
    OUTPUT_IMG_DIR = os.path.join(DATASET_DIR,'results/') #TODO: modify to match s3 bucket
    DATA_PATH = os.path.join(INPUT_IMG_DIR,'*jpg') 
    
    
    #create inference object
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=CWD)
    model.load_weights(STOMATA_WEIGHTS_PATH, by_name=True)
    
    #start inference
    files = glob.glob(DATA_PATH)
    run_inference(model, files, OUTPUT_IMG_DIR)
    
    
    return 'end of execution'

    
    

"""
SUPPORTING FUNCTIONS
"""

def stomata_filter(r_pd_):
    #stomata filer: This code filters out stomata like shapes which are of wrong size, using confidence measures.
    high_scores = r_pd_["scores"][r_pd_["scores"]>0.90]
            
    if len(high_scores) > 0:
        percentile_thres = np.min(high_scores)
    else:
        percentile_thres = np.percentile(r_pd_["scores"], 95) #st.median(r_pd_["scores"])
    
    high_conf_areas = r_pd_["areas"][r_pd_["scores"]>=percentile_thres] #conf_threshold
    high_conf_scores = r_pd_["scores"][r_pd_["scores"]>=percentile_thres]
    
    high_conf_avg_area = st.mean(high_conf_areas)
    
    above_avg = high_conf_areas[high_conf_areas>=high_conf_avg_area]
    below_avg = high_conf_areas[high_conf_areas<high_conf_avg_area]
    
    above_avg_conf = high_conf_scores[high_conf_areas>=high_conf_avg_area]
    below_avg_conf = high_conf_scores[high_conf_areas<high_conf_avg_area]
    
    #based on data length
    if len(above_avg) >= len(below_avg):
        optimal_area = np.percentile(above_avg, 50) #(st.mean(above_avg)+np.max(above_avg))/2.0 #can we use percentie 
        st_size = 'LARGE'
    else:
        #smaller elements may not be stomata, so check for their overall confidence with respect to the confidence of larger areas
        if np.mean(above_avg_conf) >= np.mean(below_avg_conf) and len(above_avg) > 1:
            optimal_area = np.percentile(above_avg, 50)
            st_size = 'LARGE'
        elif len(above_avg) <=1 and np.max(above_avg_conf) > 0.985:
            optimal_area = np.percentile(above_avg, 50)
            st_size = 'LARGE'
        else:
            optimal_area = np.percentile(below_avg, 75)
            st_size = 'SMALL'
       
    if st_size == 'LARGE':
        indices_ = r_pd_["scores"][np.logical_and(r_pd_["areas"]> (optimal_area*0.55),r_pd_["areas"]<1.5*optimal_area )].index.values.astype(int)
    else:
        indices_ = r_pd_["scores"][np.logical_and(r_pd_["areas"]> (optimal_area*0.65),r_pd_["areas"]<1.5*optimal_area )].index.values.astype(int)

    return indices_

def get_filename(string_):
    #get image filename
    folder_name = 'test/' #TODO: change accordingly when hosted on AWS. used to extract the filename
    start = string_.find(folder_name)+len(folder_name) 
    end = string_.find('.jpg',start)
    filename = string_[start:end]
    return filename


def hisEqulColor(img):
    #contrast limited histogram equalisation
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(25,25))
    channels[0] = clahe.apply(channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def sharpenColor(img):
    #sharpen image
    kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel_sharpening)
    return img

def run_inference(model, files, OUTPUT_IMG_DIR):
    
    stomata_data = pd.DataFrame(columns=['filename','num_stomata','scores','areas'])
    counter = 0
    
    for img in files:
        filename = get_filename(img)
        print(filename)
        image = cv2.imread(img)
        image = imutils.resize(image,width=1024)
        image_original = image
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #opnecv uses BGR convention
        
        
        #call histogram equalize function (optional)
        image = hisEqulColor(image)
        #image = sharpenColor(image)
        
        #convert to grayscale and save as a three channel jpeg then read it back and convert
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imwrite('current_image.jpg', image_gray)
        
        image_new = cv2.imread('current_image.jpg')
        image = cv2.cvtColor(image_new,cv2.COLOR_BGR2RGB) #opnecv uses BGR convention
        image = imutils.resize(image,width=1024)
        
        #run the image through the model
        print("making predictions with Mask R-CNN...")
        r = model.detect([image], verbose=1)[0]
        
        #create dataframe for ease of use
        r_pd = pd.DataFrame(columns=['class_id','scores', 'areas'])
            
        #create array to store areas
        num_stomata = len(r["scores"])
      
        r_pd["class_ids"] = r["class_ids"]
        r_pd["scores"] = r["scores"]
                   
        #retrieve area values from X,Y coordinates
        for i in range(0, len(r_pd["scores"])): 
            (startY,startX,endY,endX) = r["rois"][i]
            r_pd["areas"][i] = abs(startY-endY)*abs(startX-endX)
            
            
        #see how many stomata are on the image
        #1. If there are more than 2 stomata, do the following.
        #2. get the median score for confidence
        #3. get the average area for median and above
        #4. reject areas 90% or less than the average median area
        
        if num_stomata == 0:
            print("no stomata detected")
            stomata_data.loc[counter] = [filename,num_stomata,0]
            counter +=1
            cv2.imwrite(OUTPUT_IMG_DIR+filename+'_000'+'.jpg', image)
            continue
        
        if num_stomata >= 2:
            
            indices = stomata_filter(r_pd)
            #indices = r_pd["scores"][:].index.values.astype(int) #this ignores the statistical filter
    
        else: 
            indices = [0]
            
        print(indices)
                
        # loop over of the detected object's bounding boxes and masks
        areas = []
        for i in range(0, len(indices)):
            classID = r["class_ids"][indices[i]]
            mask = r["masks"][:, :, indices[i]]
            color = [0,0,0]
            areas.append(np.sum(mask == True))
    
            #uncomment to visualize the pixel-wise mask of the object
            #image = visualize.apply_mask(image, mask, color, alpha=0.5)
            
            #visualize contours
            mask[mask ==True] = 1
            mask[mask == False] = 0
            mask = mask.astype(np.uint8)
            mask,contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_original,contours, 0, [0,255,0], 4)
        
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    
        for i in range(0,len(indices)):
            (startY,startX,endY,endX) = r["rois"][indices[i]]
            classID = r["class_ids"][indices[i]]
            label = classID
            score = r["scores"][indices[i]]
            color = [255,0,0]
    
    
        id_str= str(len(indices))
        stomata_data.loc[counter] = [filename,len(indices),r["scores"][indices],areas]
        counter +=1
        cv2.imwrite(OUTPUT_IMG_DIR+filename+'_'+id_str.zfill(3)+'.jpg', image_original)
        
    
    stomata_data.to_csv(OUTPUT_IMG_DIR+'results.csv',encoding='utf-8', index=False)   


"""
SUPPORTING CLASSES
"""

class InferenceConfig(Config):
    """
    Class created for inference purposes. Extended from mrcnn.config.Config
    """
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'stomata' #provide a suitable name
    NUM_CLASSES = 1+1 #background+number of classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (12,24,48,96,192) #anchor box scales for the application
    DETECTION_MIN_CONFIDENCE = 0.6 #set min confidence threshold
    DETECTION_MAX_INSTANCES = 500
    POST_NMS_ROIS_INFERENCE = 10000
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_DIM = 800
    MEAN_PIXEL = np.array([133.774, 133.774, 133.774]) #DEFAULT VALUES FOR STOMATA MODEL. Change as necessary. #matterport takes the input as RGB 

if __name__ == '__main__':
    path = os.getcwd()
    lambda_handler(path)
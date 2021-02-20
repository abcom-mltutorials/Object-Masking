
#================================================
Copyright @ 2020 **ABCOM Information Systems Pvt. Ltd.** All Rights Reserved.

Licensed under the Apache Licaense, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.

#================================================

import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
import random
import colorsys


#MRCNN imports
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
import matplotlib.pyplot as plt


#Getting the class names
classnames = []

with open('files/coco.names') as f:
    classnames = f.read().rstrip('\n').split('\n')


#Initialising the parameters for configuring the network
class Conf(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

config = Conf()

#Creating the model
mrcnn = MaskRCNN(mode='inference',model_dir='/logs', config=config)

#Loading the weights
mrcnn.load_weights('files/mask_rcnn_coco.h5', by_name=True)


#generating a random color each segment

def random_colors_ins(N):
    
    #Generate random colors.
    #To get visually distinct colors, generating them in HSV space then
    #converting to RGB.

    #hsv = hue, saturation , brightness  
    hsv = [(i / N, 1, 1) for i in range(N)]

    #converting hsv to rgb
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    #randomly shuffling the colors
    random.shuffle(colors)
    return colors


#apply a mask over each segment

#alpha = opacity
def apply_mask_ins(image, mask, color, alpha=0.5):
#Apply the given mask to the image.
    
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


#displaying the segmented image

def display_ins(image, boxes, masks, class_ids, class_names,
                      scores=None,show_mask=True):
 
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = random_colors_ins(N)

    masked_image = image.copy()
    for i in range(N):
        color = colors[i]

        #Converting color to RGB Format
        color_rgb = [255*i for i in color]
        
        y1, x1, y2, x2 = boxes[i]

        score = scores[i] if scores is not None else None
        
        #Applying Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask_ins(masked_image, mask, color)
            
        # Label
        cv2.putText(masked_image, f"{classnames[class_ids[i]].title()} : {int(score*100)}%" ,(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,1.2,tuple(color_rgb),2)

        
    return masked_image


#for semantic segmentation

#generating random colors for different classes 
def random_colors_seg(classnames, bright=True):
    
#Generate random colors.
#To get visually distinct colors, generating them in HSV space then
#converting to RGB.
    
    brightness = 1.0 if bright else 0.7
    N = len(classnames)
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

#creating a dictionary with the different classes as keys and different colors as values for a key, using random_colors_seg for generating the different colors 
colors = {i:j for i,j in zip(classnames,random_colors_seg(classnames, bright=True))}


#apply a mask over each segment

#alpha = opacity
def apply_mask_seg(image, mask, color, alpha=0.5):
#Apply the given mask to the image.
    
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


#displaying the segmented image

def display_seg(image, boxes, masks, class_ids, class_names,
                      scores=None,show_mask=True):
 
# Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]


    masked_image = image.copy()
    for i in range(N):

        color_rgb = [255*i for i in colors[classnames[class_ids[i]]]]
        
        y1, x1, y2, x2 = boxes[i]

        score = scores[i] if scores is not None else None
        
    # Label
        cv2.putText(masked_image, f"{classnames[class_ids[i]].title()} : {int(score*100)}%" ,(x1,y1-5),cv2.FONT_HERSHEY_PLAIN,1.2,tuple(color_rgb),2)

    # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask_seg(masked_image, mask, colors[classnames[class_ids[i]]])

        
    return masked_image


#detecting objects

inp = int(input('Choose the type of Image Segmentation : \n 1.Instance Segmentation \n 2.Semantic Segmentation \n'))

if inp == 1: #for instance
    inp1 = int(input('Choose the format for detecting objects : \n 1.Image \n 2.Video \n 3.Webcam \n'))

    if inp1 == 1: #for image
        img = cv2.imread('data/image00.jpg')

        #Information about the objects detected the image
        results_list = mrcnn.detect([img], verbose=1)
        results = results_list[0]

        image = display_ins(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])

        cv2.imshow('Image',image)
        cv2.waitKey(0)

    elif inp1 == 2: #for video
        cap = cv2.VideoCapture('data/video00.mp4')

        while True:
            success, img = cap.read()
    
            results = mrcnn.detect([img], verbose=0)[0]
    
            img = display_ins(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])
    
            cv2.imshow('Vid',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    
    elif inp1 == 3: #for webcam
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
    
            results = mrcnn.detect([img], verbose=0)[0]
    
            img = display_ins(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])
    
            cv2.imshow('Vid',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

elif inp == 2: #for semantic
    inp1 = int(input('Choose the format for detecting objects : \n 1.Image \n 2.Video \n 3.Webcam \n'))

    if inp1 == 1: #for image
        img = cv2.imread('data/image00.jpg')

    #Information about the objects detected the image
        results_list = mrcnn.detect([img], verbose=1)
        results = results_list[0]

        image = display_seg(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])

        cv2.imshow('Image',image)
        cv2.waitKey(0)

    elif inp1 == 2: #for video
        cap = cv2.VideoCapture('data/video00.mp4')

        while True:
            success, img = cap.read()
    
            results = mrcnn.detect([img], verbose=0)[0]
    
            img = display_seg(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])
    
            cv2.imshow('Vid',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    elif inp1 == 3: #for webcam
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
    
            results = mrcnn.detect([img], verbose=0)[0]
    
            img = display_seg(img, results['rois'], results['masks'], results['class_ids']-1, classnames, results['scores'])
    
            cv2.imshow('Vid',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

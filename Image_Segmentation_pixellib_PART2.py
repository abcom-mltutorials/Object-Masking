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
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.semantic import semantic_segmentation


#Semantic Segmentation
semantic_segment = semantic_segmentation()

#we are going to use deeplabv3_xception model for ade20k dataset here
semantic_segment.load_ade20k_model("files/deeplabv3_xception65_ade20k.h5")


#Instance Segmentation
inst_segment = instance_segmentation()

#we are going to use the mask_rcnn for coco dataset here
inst_segment.load_model("files/mask_rcnn_coco.h5")

inp = int(input('Choose the type of Image Segmentation : \n 1.Instance Segmentation \n 2.Semantic Segmentation \n'))

if inp == 1: #for instance
    inp1 = int(input('Choose the format for detecting objects : \n 1.Image \n 2.Video \n 3.Webcam \n'))

    if inp1 == 1: #for image
        img_ins_seg = inst_segment.segmentImage("data/image00.jpg")

    
        #Uncomment next two lines if you want to see the original image
        #cv2.imshow("Image",cv2.imread('data/image00.jpg'))
        #cv2.waitKey(0)

        #Showing the image with segmentations
        cv2.imshow('Image',img_ins_seg[1])
        cv2.waitKey(0)

    elif inp1 == 2: #for video
        inst_segment.process_video("data/video00.mp4", frames_per_second= 50, output_video_name="video_inst.mp4")

        #Playing the video
        vid = cv2.VideoCapture('video_inst.mp4')

        while True:
            success, img = vid.read()
    
            cv2.imshow('Segmented Video',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    elif inp1 == 3: #for webcam
        capture = cv2.VideoCapture(0)

        inst_segment.process_camera(capture, show_bboxes = True, frames_per_second = 15, output_video_name = "webcam_inst.mp4", show_frames = True, frame_name = "frame")


elif inp == 2: #for semantic
    inp1 = int(input('Choose the format for detecting objects : \n 1.Image \n 2.Video \n 3.Webcam \n'))

    if inp1 == 1: #for image
        img_sem_seg = semantic_segment.segmentAsAde20k("data/image00.jpg", overlay = True)


        #Uncomment next two lines if you want to see the original image
        #cv2.imshow("Image",cv2.imread('data/image00.jpg'))
        #cv2.waitKey(0)

        #Showing the image with segmentations
        cv2.imshow("Image",img_sem_seg[1])
        cv2.waitKey(0)

    elif inp1 == 2: #for video
        semantic_segment.process_video_ade20k("data/video00.mp4", frames_per_second= 50, output_video_name="video_seg.mp4")


        #Playing Video
        vid = cv2.VideoCapture('video_seg.mp4')

        while True:
            success, img = vid.read()
    
            cv2.imshow('Segmented Video',img)
    
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    elif inp1 == 3: #for webcam
        capture = cv2.VideoCapture(0)

        semantic_segment.process_camera_ade20k(capture, overlay=True, frames_per_second= 15, output_video_name="webcam_seg.mp4", show_frames= True, frame_name= "frame", check_fps = True)
        

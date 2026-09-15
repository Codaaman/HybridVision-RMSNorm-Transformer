import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch as t
import torch.nn as nn
import cv2 as cv
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import mediapipe as mps
import mediapipe.tasks as mp
import os
import threading
from Base_Model import imageblock
from PIL import Image
import time


device=t.device("cuda")




Model=imageblock().to(device=device)
Model_static=t.load(r"D:\Torch_frame_work\personal_vision_2.pth",weights_only=False,map_location=device)
Model.load_state_dict(Model_static)

        
Model.eval()
trns=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])


def my_model(img):

    #img = Image.open(img).convert('RGB')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = trns(img)

    with t.no_grad():
        img=image.unsqueeze(0).to(device)
        result=Model(img)
        threshold=t.softmax(result,dim=1)
        thres,r=t.max(threshold,1)
        print(r.item(), thres.item()*100)
        if r.item()==0 and thres.item()*100>=82:
            return "Aman"
        elif r.item()==1 and thres.item()*100>=82:
            return "Mummy"
        else:
                return "unknown"

        








def speed(d):
    
    if d>80:
        print("SLOW")
    elif 10<d:
        print("FAST")
        
    

    
model=mp.BaseOptions
dector=mp.vision.FaceDetector
options=mp.vision.FaceDetectorOptions
VisionRunningMode = mp.vision.RunningMode

a=cv2.VideoCapture(0)

options=options(base_options=model(model_asset_path=r'D:\Torch_frame_work\COMPUTER_VISION\detector_full_range.tflite'),running_mode=VisionRunningMode.VIDEO)

frame=0
with dector.create_from_options(options) as detector: 
    while True:
        

        rat,img=a.read()

        if not rat:
            break

        w_img,h_img,_=img.shape
        co_img=mps.Image(image_format=mps.ImageFormat.SRGB,data=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        timestamp_ms = int(a.get(cv2.CAP_PROP_POS_MSEC))
        result = detector.detect_for_video(co_img, timestamp_ms)
        frame+=1
        if result.detections:
            
            for i,detection in enumerate(result.detections):
                bbox=detection.bounding_box
                x=bbox.origin_x
                y=bbox.origin_y

                w = bbox.width
                h = bbox.height

                real_size=h*w
                image_size=img.shape[0]*img.shape[1]
                focal_length=720

                dis=(real_size*focal_length)/image_size
               
                padding_ratio = 0.55 
                pad_w = int(w * padding_ratio)
                pad_h = int(h * padding_ratio)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)

                x2 = min(w_img, x + w + pad_w)
                y2 = min(h_img, y + h + int(pad_h * 1.5)) 
                crop_img = img[y1:y2,x1:x2]

                label=my_model(crop_img)


                if label is None:
                    print(label)
                    padding_ratio+=0.1
                elif label is not None:
                    padding_ratio-=0.1
                    
                    

                cv2.putText(img,label,(x, y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)
                cv2.rectangle(img,(x, y), (x + w, y + h), (0, 255, 0), 2)

        
            
        cv2.imshow("win",crop_img)

        c=cv2.waitKey(1)

        if c==ord("q"):
            break
               


    
    
    






    



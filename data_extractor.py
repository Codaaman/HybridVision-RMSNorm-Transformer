# from huggingface_hub import login
# import os
# os.environ["HF_HOME"]=r"D:\Torch_frame_work\COMPUTER_VISION\facedata"
# os.environ["HF_HUB_CACHE"] = r"D:\Torch_frame_work\COMPUTER_VISION\facedata\hub"
# os.environ["HF_DATASETS_CACHE"] = r"D:\Torch_frame_work\COMPUTER_VISION\facedata\datasets"
# os.environ["HF_XET_HIGH_PERFORMANCE"]="1"
# os.environ["HF_XET_NUM_CONCURRET_RANGE_GETS"]="16"
# from datasets import load_dataset
#from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd

#import cv2 as cv
#import time



#dataset=ImageFolder(r"D:\Torch_frame_work\COMPUTER_VISION\facedata\hub\datasets--AI-Solutions-KK--face_recognition_dataset\snapshots\98adc28fb357261641db208b377e0946ba7369c8\face_recognition_dataset")











# trns_for_training=transforms.Compose([transforms.Resize((244,244)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(20),transforms.ToTensor()])
# trns_for_validation=transforms.Compose([transforms.Resize((244,244)),transforms.ToTensor()])


# def apply_trnsform1(batch):
#     batch['image']=[trns_for_training(img) for img in batch["image"]]
#     return batch

# def apply_trnsform2(batch):
#     batch['image']=[trns_for_validation(img) for img in batch["image"]]
#     return batch




# def collate_fn():
#     #snapshot_download(repo_id="chronopt-research/cropped-vggface2-224",repo_type="dataset",tqdm_class=tqdm)
#     #dataset = load_dataset( "chronopt-research/cropped-vggface2-224")
#     #dataset = dataset["train"].select(range(8173))  # Select the first 1000 samples for training 

#     dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

#     dataset1=dataset["train"]                              
#     dataset2=dataset["test"]

#     dataset1=dataset1.with_transform(apply_trnsform1)
#     dataset2=dataset2.with_transform(apply_trnsform2)

#     img_data=DataLoader(
#             dataset1,
#             batch_size=32,
#             shuffle=True,

#             )

#     validation=DataLoader(
#             dataset2,
#             batch_size=32,
#             shuffle=True,
#             )


#     return img_data,validation

    




def data_collection():
    l1=[]
    l2=[]

    trns1=transforms.Compose([transforms.Resize((244, 244)),
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.RandomRotation(degrees=15),
                     transforms.ColorJitter(brightness=0.3, contrast=0.3),
                     transforms.ToTensor()])
    trns2=transforms.Compose([transforms.Resize((244,244)),transforms.ToTensor()])

    
    
    data1=ImageFolder(r"D:\Torch_frame_work\COMPUTER_VISION\archive (2)\splited_105_face_recognition\train",transform=trns1)
    data2=ImageFolder(r"D:\Torch_frame_work\COMPUTER_VISION\archive (2)\splited_105_face_recognition\test",transform=trns2)

    training=DataLoader(data1,shuffle=True,batch_size=32)
    validation=DataLoader(data2,shuffle=True,batch_size=32)

    # train=int(0.8*len(data))
    # test=len(data)-train

    # Genrator=torch.Generator().manual_seed(42)
    # print("going to split.....","\n","v")
    # traing,validation=random_split(data,[train,int(test)],generator=Genrator)
    # print("trnsformed for training ...","\n","v")
    # training=DataLoader([(trns1(x[0]),x[1]) for x in traing],shuffle=True,batch_size=16)
    # validation=DataLoader([(trns2(x[0]),x[1]) for x in validation],shuffle=True,batch_size=16)
    # print("now loading ....","\n","v")


    return training,validation



    # print("data is appending in list...","\n","v")
    # for k in data:
    #     l1.append(k[0])
    #     l2.append(k[1])



    # print("converting into data_frame....","\n","v")
    # d=pd.DataFrame({"img":l1,"label":l2})
    # X=np.array(d["img"])
    # Y=np.array(d["label"])
    # print("going to split.....","\n","v")
    # x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=42)



    # x_tr=[]
    # x_te=[]


    # print("trnsformed for training ...","\n","v")
    # for x in x_train:
    #     x_tr.append(trns(x))

    # for x in x_test:
    #     x_te.append(trns2(x))
    
            


    # print("now loading ....","\n","v")
    # test=DataLoader([(x,y) for x,y in zip(x_tr,y_train)],shuffle=True,batch_size=16)
    # validator=DataLoader([(x,y) for x,y in zip(x_te,y_test)],shuffle=True,batch_size=16)



    #return test,validator



if __name__=="__main__":
     data_collection()

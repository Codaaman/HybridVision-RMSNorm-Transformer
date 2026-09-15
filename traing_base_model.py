import torch
from face_recognization import imageblock
import torch.nn as nn
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


def data_collection():
    l1=[]
    l2=[]

    trns=transforms.Compose([transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()])
    trns2=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    
    
    data=ImageFolder(r"D:\Torch_frame_work\COMPUTER_VISION\images")
    
    for k in data:
        l1.append(k[0])
        l2.append(k[1])



    d=pd.DataFrame({"img":l1,"label":l2})
    X=np.array(d["img"])
    Y=np.array(d["label"])
    x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.8,random_state=42)

    x_tr=[]
    x_te=[]


    for x in x_train:
        x_tr.append(trns(x))
    


    for x in x_test:
            x_te.append(trns2(x))
    
            


   
    test=DataLoader([(x,y) for x,y in zip(x_tr,y_train)],shuffle=True,batch_size=16)
    validator=DataLoader([(x,y) for x,y in zip(x_te,y_test)],shuffle=True,batch_size=16)



    return test,validator


    










device=torch.device("cuda")

model=imageblock().to(device)
model_stat=torch.load("vision_version_1.pth",weights_only=False)
model.load_state_dict(model_stat)



for perm in model.parameters():
    perm.requires_grad=False

# for param in model.classifier.parameters():
#     param.requires_grad = True  

model.classifier=nn.Linear(128,3).to(device)



optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
sheduler=CosineAnnealingLR(optimizer,T_max=20)
loss=nn.CrossEntropyLoss().to(device)
train,validate=data_collection()
epoch=20


for k in range(epoch):
    c=0
    total_loss=0
    traing_loss=0
    val_loss=0
    total_corect=0
    valid_total=0

    model.train()
    for img,label in train:
        print(f"\rnumber of images:{c+1}/{len(train)}",end=" ",flush=True)
        optimizer.zero_grad()
        result=model(img.to(device))
        target=label.long()
        loss1=loss(result,target.to(device))
        loss1.backward()
        optimizer.step()
        total_loss+=loss1.item()
        c+=1
    total_loss/=len(train)
    model.eval()
    with torch.no_grad():
        for img,label in validate:
            result=model(img.to(device))
            target=label.long()

            loss2=loss(result,target.to(device))
            val_loss+=loss2.item()
            valid_total+=target.size(0)
            _,predict=torch.max(result,1)
            total_corect+=(predict==target.to(device)).sum().item()
    val_loss/=len(validate)
    validation_acuracy=100*total_corect/valid_total
    sheduler.step()


    print(f" epch : {k+1}  train_loss: {total_loss:4f} validation_loss: {val_loss:4f} validation_acuracy: {validation_acuracy:.2f}%")

    if (validation_acuracy == 100.0) or (validation_acuracy == 98.0):
        print("best_accuracy")
        break

torch.save(model.state_dict(),r"d:\python programing\game_automation\Assistance_with_recognation\personal_vision_2.pth") 
print("model_save")     







    
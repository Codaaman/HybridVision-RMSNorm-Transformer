import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch as t
import torch.nn as nn
import cv2 as cv
import data_extractor as dt
import numpy as np
from tqdm import tqdm
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau
import gc
device=t.device("cuda")


class Block1(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)
        )


        
    def forward(self,x):
        #x=self.embed(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp


        return x
    


class Block2(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)
        )

    

        
    def forward(self,x):
        #x=self.embed(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp

    

        return x
    


class Block3(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)
        )

    

        
    def forward(self,x):
        #x=self.embed(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp



        return x



class Block4(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)
        )


        
    def forward(self,x):
        #x=self.embed(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp

    

        return x
    

class Block5(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)
        )

        

        
    def forward(self,x):
        #x=self.embed(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp

        

        return x    




class main_Block(nn.Module):
    def __init__(self):
        super().__init__()

        #self.embed=nn.Embedding(embedding_dim=512)
        self.rms1=nn.RMSNorm(128)
        self.attention=nn.MultiheadAttention(128,8,batch_first=True)
        self.rms2=nn.RMSNorm(128)
        self.mlp=nn.Sequential(
            nn.Linear(128,1024),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(1024,128),
            nn.Dropout(0.2)

        )

    

        self.Block1=Block1()    
        self.Block2=Block2()    
        self.Block3=Block3()    
        self.Block4=Block4()    
        self.Block5=Block5()    
        
    def forward(self,x):
        x=self.Block1(x)
        x=x+self.Block2(x)
        x=x+self.Block3(x)
        x=x+self.Block4(x)
        x=x+self.Block5(x)
        atten,_=self.attention(self.rms1(x),self.rms1(x),self.rms1(x))
        x=x+atten
        mlp=self.mlp(self.rms2(x))
        x=x+mlp


        return x




class imageblock(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed=nn.Embedding(196,128)

        self.cnn1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.cnn11=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.batch_norm1=nn.BatchNorm2d(64)
        self.rlu1=nn.ReLU()
        self.maxpool_1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop1=nn.Dropout(0.2)

        self.cnn2=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.cnn22=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.batch_norm2=nn.BatchNorm2d(128)
        self.rlu2=nn.ReLU()
        self.maxpool_2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.drop2=nn.Dropout(0.2)

        self.cnn3=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.cnn33=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.batch_norm3=nn.BatchNorm2d(128)
        self.rlu3=nn.ReLU()
        self.maxpool_3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop3=nn.Dropout(0.2)

        self.cnn4=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.cnn44=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.batch_norm4=nn.BatchNorm2d(128)
        self.rlu4=nn.ReLU()
        self.maxpool_4=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop4=nn.Dropout(0.2)

        self.final_drop=nn.Dropout(0.2)
        self.classifier = nn.Linear(128, 105)
        self.activation=nn.GELU()

        self.main_block=main_Block()


    def forward(self,x):
         x=self.cnn1(x)
         x=self.cnn11(x)
         x=self.batch_norm1(x)
         x=self.rlu1(x)
         x=self.maxpool_1(x)
         x=self.drop1(x)

         x=self.cnn2(x)
         x=self.cnn22(x)
         x=self.batch_norm2(x)
         x=self.rlu2(x)
         x=self.maxpool_2(x)
         x=self.drop2(x)

         x=self.cnn3(x)
         x=self.cnn33(x)
         x=self.batch_norm3(x)
         x=self.rlu3(x)
         x=self.maxpool_3(x)
         x=self.drop3(x)
        #  img=x.detach().cpu().numpy()
        #  img=img[0,0,:,:]
         
        #  img=cv.normalize(img,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX)
        #  img=img.astype(np.uint8)
        #  img= cv.resize(img, (400, 400), interpolation=cv.INTER_NEAREST)
        #  cv.imshow("image_1",img)
        #  c=cv.waitKey(1)
        #  if c==ord('q'):
        #      cv.destroyAllWindows()
         

         x=self.cnn4(x)
         x=self.cnn44(x)
         x=self.batch_norm4(x)
         x=self.rlu4(x)
         x=self.maxpool_4(x)
         x=self.drop4(x)

         x = nn.AdaptiveAvgPool2d((14,14))(x)
         x=x.flatten(2)
         x=x.transpose(1,2)

         x=x+self.pos_embed(t.arange(x.size(1)).to(device))

         x=self.main_block(x)

         #x=self.activation(x)

         x = x.mean(dim=1) 
         x=self.final_drop(x)          

         x = self.classifier(x)

         return x
        

















traing_data,validation_data=dt.data_collection()
if __name__=="__main__":

    model=imageblock().to(device)
    print(next(model.parameters()).device)
    optimizer=t.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=5e-4,)
    sheduler=CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-8)
    #sheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=2)
    print("optimizer created")
    loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    print("loss function created")
    


    scaler=t.amp.GradScaler("cuda")
    for epoch in range(100):
        
        model.train()
        total_corect=0
        valid_total=0
        total_loss=0
        val_loss=0
        c=1
        for image,label in traing_data :
                print(f"\rbatch : {c}/{len(traing_data)}",end="",flush=True)
                img=image.to(device)
                

                target = label.long()

                optimizer.zero_grad()
                with t.amp.autocast("cuda"):
                    result=model.forward(img)
                    loss1=loss_fn(result,target.to(device))
                    
                scaler.scale(loss1).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss+=loss1.item()
                c+=1
        total_loss/=len(traing_data)
        model.eval()
        
        with t.no_grad():
            for image,label in validation_data:
                img=image.to(device)
                target = label.long()     
                result=model.forward(img)
        

                loss=loss_fn(result,target.to(device))

                val_loss+=loss.item()
                valid_total+=target.size(0)
                _,predict=t.max(result,1)

                total_corect+=(predict==target.to(device)).sum().item()
        val_loss/=len(validation_data)
        validation_acuracy=100*total_corect/valid_total
        sheduler.step()
        current_lr = sheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[0]['lr']

        print(f" epoch:{epoch+1} validataion_loss: {val_loss:4f} traing_loss: {total_loss:4f} validation_score: {validation_acuracy:.2f}   learning_rate: {current_lr:.2e}")

        t.cuda.empty_cache()
        gc.collect()
                    
    t.save(model.state_dict(),"vision_version_1.pth")








    



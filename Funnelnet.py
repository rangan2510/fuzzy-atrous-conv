#Installs
#!pip install -U fvcore
# !pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Imports and Initialize
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean

#usrlibs
from modelhistory import ModelHistory
from howlong import HowLong

#torch
import torch 
import torch.nn as nn
from torch.nn import Conv2d #, ChannelShuffle
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

total_howlong = HowLong()

DIM = 512 # SET IMAGE DIMENSION 



# Hyper parameters
num_epochs = 1
num_classes = 2
batch_size = 10
learning_rate = 0.0005

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)



lines = []
train_data=[]
test_data=[]
with open('../input/covidx-cxr2/train.txt') as f:
    lines = f.readlines()

count = 0
for line in lines:
    l=line.split()
    label=0
    if (l[2]=='positive'):
        label=1
    train_data.append(["../input/covidx-cxr2/train/"+l[1],label])

lines = []
with open('../input/covidx-cxr2/test.txt') as f:
    lines = f.readlines()

count = 0
for line in lines:
    l=line.split()
    label=0
    if (l[2]=='positive'):
        label=1
    test_data.append(["../input/covidx-cxr2/test/"+l[1],label])

images_df = pd.DataFrame(data=train_data, columns=["images", "labels"])
#print(images_df.head(10))
#images_df.groupby('labels').size()



# handling imbalance
positive_df = images_df[images_df['labels'] == 1]
frames = [images_df, positive_df,positive_df,positive_df,positive_df,positive_df]
#images_df = pd.concat(frames)
#images_df.groupby('labels').size()



test = pd.DataFrame(data=test_data, columns=["images", "labels"])
#test.groupby('labels').size()



train, val = train_test_split(images_df, stratify=images_df.labels, test_size=0.015)
#len(train),  len(val), len(test)



class MyDataset(Dataset):
    def __init__(self, df_data,transform=None):
        super().__init__()
        self.df = df_data.values
        
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path,label = self.df[index]
        
        image = cv2.imread(img_path)
        image = cv2.resize(image, (DIM,DIM))
        if self.transform is not None:
            image = self.transform(image)
        return image, label



hist = ModelHistory()



trans_train = transforms.Compose([transforms.ToPILImage(),
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.RandomHorizontalFlip(), 
                                  transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(20), 
                                  transforms.Resize(DIM, interpolation = 2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

trans_valid = transforms.Compose([transforms.ToPILImage(),                    
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.Resize(DIM, interpolation = 2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

dataset_train = MyDataset(df_data=train, transform=trans_train)
dataset_valid = MyDataset(df_data=val,transform=trans_valid)
dataset_test = MyDataset(df_data=test,transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
loader_test = DataLoader(dataset = dataset_test, batch_size=batch_size//2, shuffle=False, num_workers=0)



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss



import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import math

class CustomConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, dilation=2, padding=0, stride=2, bias=True, mu=0.1):
        super(CustomConv,self).__init__()
        device = "cpu"
        if(torch.cuda.is_available()):
            device = 'cuda'
            
        self.kernel_size=_pair(kernel_size)
        self.out_channels=out_channels
        self.dilation=_pair(dilation)
        self.padding=_pair(padding)
        self.stride=_pair(stride)
        self.in_channels=in_channels
        self.mu=mu
        self.mu_=(1-mu)/3
        self.bias1=torch.nn.Parameter(torch.Tensor(out_channels))
        self.device = device
        
        #self.bias1=self.bias1.to(device)
        mu_=self.mu_
        self.calculated_kernel_size=self.dilation[0]*(self.kernel_size[0]-1)+1
        self.weight=torch.nn.Parameter(torch.Tensor(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1]))
        self.fuz=self.mask_dial(self.kernel_size[0],self.dilation[0],self.mu)
        self.fuz[self.fuz==1] = 0
        self.fuz=self.fuz.unsqueeze(0)#.unsqueeze(0).unsqueeze(0)
        
        temp=self.fuz
        for i in range(1,self.in_channels):
            temp=torch.cat((temp,self.fuz))
        temp=temp.unsqueeze(0)
        temp1=temp
        for i in range(1,self.out_channels):
            temp1=torch.cat((temp1,temp))
        self.fuz=temp1
        if(bias):
            self.bias=torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameters("bias",None)
        self.fuz=self.fuz.to(self.device)
        self.reset_parameters()
    
    def mask_dial(self,kernel_size,dilation,mu):
        dilation-=1
        mid=[0 for i in range(dilation)]
        lim=(dilation//2) if (dilation%2==0) else ((dilation//2)+1)
        diff=(1-mu)/lim
        filter1=[]
        for i in range(lim):
            mid[i]=1-(i+1)*diff
            mid[dilation-1-i]=1-(i+1)*diff
        for i in range(2*kernel_size-1):
            if(i%2==0):
                filter1=filter1+[0]
            else:
                filter1=filter1+mid
        filter2=[[0 for i in range(dilation+2)] for j in range(dilation)]
        for i in range(lim):
            for j in range(i+2):
                filter2[i][j]=mid[i]
                filter2[i][dilation+1-j]=mid[i]
                filter2[dilation-i-1][j]=mid[i]
                filter2[dilation-i-1][dilation+1-j]=mid[i]
            for j in range(i+1,lim):
                filter2[i][j+1]=mid[j]
                filter2[i][dilation-j]=mid[j]
                filter2[dilation-i-1][j+1]=mid[j]
                filter2[dilation-i-1][dilation-j]=mid[j]
        filter3=[x[1:] for x in filter2]
        for i in range(kernel_size-2):
            for j in range(len(filter2)):
                filter2[j]+=filter3[j]
        result=[]
        for i in range(2*kernel_size-1):
            if(i%2==0):
                result=result+[filter1]
            else:
                result=result+filter2
                result=[0 for i in range(2*kernel_size-1)]
        result=[]
        for i in range(2*kernel_size-1):
            if(i%2==0):
                result=result+[filter1]
            else:
                result=result+filter2
        result=torch.Tensor(result)
        return result

    
    def reset_parameters(self):
        stdv=math.sqrt(6./((self.in_channels*(self.kernel_size[0]**2))+(self.out_channels*(self.kernel_size[0]**2))))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
        self.bias1.data.uniform_(-stdv,stdv)
    
    
    def forward(self, input_):

        hout = ((input_.shape[2]+2*self.padding[0]-self.calculated_kernel_size)//self.stride[0])+1
        wout = ((input_.shape[3]+2*self.padding[1]-self.calculated_kernel_size)//self.stride[1])+1
               
        weight_kernel = F.unfold(input_,kernel_size=self.kernel_size,dilation=self.dilation,stride=self.stride).to(self.device)
        fuzzy_kernel = F.unfold(input_,kernel_size=self.calculated_kernel_size,dilation=1,stride=self.stride).to(self.device)
        
        convolvedOutput = (fuzzy_kernel.transpose(1,2).matmul((((self.fuz.permute(1,2,3,0))*self.bias1).permute(3,0,1,2)).flatten(1).transpose(0,1))).transpose(1,2)+(weight_kernel.transpose(1,2).matmul((self.weight).flatten(1).transpose(0,1))).transpose(1,2)
        convolutionReconstruction=convolvedOutput.view(input_.shape[0],self.out_channels,hout,wout)
        return convolutionReconstruction



#Top of the funnel
class TOFU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d( in_channels, 16, kernel_size=3, dilation=1, padding=1, padding_mode='replicate' )
        self.conv2 = Conv2d( 16,32, kernel_size=3, dilation=3, padding=3, padding_mode='replicate')
        self.conv3 = Conv2d(16+32,48, kernel_size=3, dilation=5, padding=5, padding_mode='replicate')
        
        self.compress = Conv2d(16+32+48,out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(torch.cat([x1,x2],1)))
        x4 = F.relu(self.compress(torch.cat([x1,x2,x3],1)))
        x4 = self.bn(x4)
        return x4
        
#Middle of the Funnel
class MOFU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fconv1 = CustomConv(in_channels,16,kernel_size=3, stride=2,dilation=2, mu=0.1)
        self.fconv2 = CustomConv(16,32,kernel_size=3, stride=3,dilation=3, mu=0.3)
        self.fconv3 = CustomConv(32,out_channels,kernel_size=3, stride=5,dilation=5, mu=0.5)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.fconv1(x))
        x = F.relu(self.fconv2(x))
        x = F.relu(self.fconv3(x))
        x = self.bn(x)
        return x
        

import cv2
import matplotlib.pyplot as plt

class Net(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()
        self.tofu = TOFU(3,32)
        self.mofu = MOFU(32,32)
        self.fc = nn.Linear(32*15*15, num_classes) 
        self.gradients = None
    def activations_hook(self, grad):
        self.gradients = grad
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)
    def forward(self, x):
        x = self.tofu(x)
        x = self.mofu.fconv1(x)
        x = self.mofu.fconv2(x)
        x = self.mofu.fconv3(x)
        h = x.register_hook(self.activations_hook)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

net = Net(num_classes)
model=net.to(device)
criterion = FocalLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)




import torch.nn as nn
from torchvision import models
model = net.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Total trainable params:", pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in model.parameters())
#print("All params:", pytorch_total_params)



# Loss and optimizer
criterion = FocalLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.0002)



model.load_state_dict(torch.load("../input/covid-cxr-3-layers/final_state1.dct"))


print("Enter image path\n")
img_path=input()

#creating gradcam image

image = cv2.imread(img_path)
image = cv2.resize(image, (DIM,DIM))
trans = transforms.Compose([transforms.ToPILImage(),                    
                                  transforms.Pad(64, padding_mode='reflect'),
                                  transforms.Resize(DIM, interpolation = 2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
img = trans(image).to(device)
pred = model(img.unsqueeze(0))#.argmax(dim=1)
pred[:,torch.argmax(pred)].backward()
gradients = net.get_activations_gradient()
gradients.shape
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
activations = net.tofu(img.unsqueeze(0))
activations = net.mofu.fconv1(activations)
activations = net.mofu.fconv2(activations)
activations = net.mofu.fconv3(activations).detach()
activations.shape
for i in range(32):
    activations[:, i, :, :] *= pooled_gradients[i]
heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap.cpu(), 0)
heatmap /= torch.max(heatmap)
plt.matshow(heatmap.squeeze())
img=img.permute(1,2,0)
heatmap = cv2.resize(heatmap.numpy(), (img.shape[0], img.shape[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.1 + img.cpu().numpy()
cv2.imwrite('./map.jpg', superimposed_img)
#plt.imshow( heatmap * 0.002 + img.cpu().numpy()  )
#save gradcam and original image
x=heatmap * 0.002 + img.cpu().numpy()
y=np.transpose(x, axes=[2,0,1])
y=torch.tensor(y)
from torchvision.utils import save_image
save_image(y, 'gradcam1.png')
img=img.permute(2, 0, 1)
#save_image(img, 'original1.png')
pred1=""
if(pred==1):
    pred1="COVID"
else:
    pred1="Normal"
print("Predicted class:",pred1)
print("Gradcam image saved at ./gradcam1.png")
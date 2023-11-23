
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F

import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import pickle

assert torch.cuda.is_available(), "GPU is not enabled"




device = torch.device('cuda')

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



path_ge = '/export/fhome/mapsiv/QuironHelico/CroppedPatches'


dataset=pd.read_csv(path_ge+'/metadata.csv')
nou=dataset.loc[dataset['DENSITAT'] == 'NEGATIVA']
llista_carpetes=os.listdir(path_ge)

carpetes_negatives=[]
for carpeta in llista_carpetes:
    if '.' not in carpeta:
      #print(carpeta)
      nom=carpeta.split("_")[0]
      if nom in list(nou["CODI"]):
          carpetes_negatives.append(os.path.join(path_ge,carpeta))
      
nom_imagenes=[]

nom_imagenes_test=[]
"""  
for carpeta in carpetes_negatives:
    llistat=[]
    for element in os.listdir(path+'/'+ carpeta):#canviar path en cluster 
        nom_imagenes.append(carpeta+'/'+element)
"""
total_pacients_sans=len(carpetes_negatives)
print('total_pacients_sans',total_pacients_sans)

num_cops=0
for carpeta_path in carpetes_negatives[:50]:
  num_cops+=1
  
  path_local=os.path.join(path_ge,carpeta_path)
  aux=[os.path.join(path_local,x) for x in (os.listdir(carpeta_path))]
  nom_imagenes= nom_imagenes +aux#[0::10]
  
  
num_cops_test=0
for carpeta_path_test in carpetes_negatives[30:35]:
  num_cops_test+=1
  
  path_local_test=os.path.join(path_ge,carpeta_path_test)
  aux=[os.path.join(path_local_test,x) for x in (os.listdir(carpeta_path_test))]
  nom_imagenes_test= nom_imagenes_test +aux#[0::10]
  
#nom_imagenes_2=os.listdir(path_2)
#nom_imagenes=nom_imagenes+nom_imagenes_2

print('num_cops',num_cops)
print('len(nom_imagenes)',len(nom_imagenes))
print('num_cops_test',num_cops_test)
print('len(nom_imagenes_test)',len(nom_imagenes_test))
#train,test=train_test_split(nom_imagenes, test_size=0.3)



DIM=(64,64)
batch_size=512

#ni idea del maximo
#183W / 250W amb 128
#7010MiB / 12288MiB amb 160
ll_img_train=[]
for titulo in nom_imagenes:
  img =  cv.imread(os.path.join(path_ge,titulo))
  img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
  ll_img_train.append(cv.resize(img, DIM, interpolation=cv.INTER_AREA))
  #ll_img_train.append(cv.resize(img, None, fx=1, fy=1, interpolation=cv.INTER_AREA))


ll_img_test=[]
for titulo in nom_imagenes_test:
  img =  cv.imread(os.path.join(path_ge,titulo))
  img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
  ll_img_test.append(cv.resize(img, DIM, interpolation=cv.INTER_AREA))  
  #ll_img_test.append(cv.resize(img, None, fx=1, fy=1, interpolation=cv.INTER_AREA))  

loader_train= torch.utils.data.DataLoader( ll_img_train, batch_size=batch_size, shuffle=True ) #, pin_memory=True
loader_test = torch.utils.data.DataLoader( ll_img_test , batch_size=batch_size, shuffle=True ) #, pin_memory=True


print("loader_train")
n_batches=0
for batch_features in loader_train:

    n_batches+=1
print('batch_features.size()',batch_features.size())
print('n_batches',n_batches)

print("loader_test")
n_batches=0
for batch_features in loader_test:

    n_batches+=1
print('batch_features.size()',batch_features.size())
print('n_batches',n_batches)

print("a punto de hacer convae")



class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # Nueva capa
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1),
        )
        
  
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = ConvAE()


print(model)


os.makedirs('imagenes_originales_train', exist_ok=True)
os.makedirs('imagenes_reconstruidas_train', exist_ok=True)

os.makedirs('imagenes_originales_test', exist_ok=True)
os.makedirs('imagenes_reconstruidas_test', exist_ok=True)

def train(model, loader, optimizer, criterion, device):
    loss = 0
    model.train()
    losses =list()
    img_count = 0
    for batch_features in loader:
       
  
       
        batch_features = batch_features.float() / 255.0
        batch_features = batch_features.permute(0, 3, 1, 2)  
        
        batch_features = batch_features.to(device)
        
        optimizer.zero_grad()
       
        outputs = model(batch_features)
        
        outputs = F.interpolate(outputs, size=DIM, mode='bilinear', align_corners=False)
      
        train_loss = criterion(outputs, batch_features)
      
        train_loss.backward()
       
        optimizer.step()

        loss += train_loss.item()
        

    
    loss = loss / len(loader)
    print(f"Train loss = {loss:.6f}")
    losses.append(loss)
       
    return loss
    
    
def train_print(model, loader, optimizer, criterion, device):
    loss = 0
    model.train()
    losses =list()
    img_count = 0
    for batch_features in tqdm(loader):
        batch_features = batch_features.float() / 255.0
        batch_features = batch_features.permute(0, 3, 1, 2)  
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        #print(batch_features.size())
        #print(outputs.size())
        outputs = F.interpolate(outputs, size=DIM, mode='bilinear', align_corners=False)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()
        

        #print("Imatge original")
        img_original = batch_features.detach().cpu().numpy()
        plt.close() 
        plt.figure()
        plt.axis("off")
        
        plt.imshow(img_original[0].transpose(1, 2, 0))
        plt.savefig(f'imagenes_originales_train/img_original_{img_count}.png')
        plt.close() 
       
        #print("Imatge reconstruida")
        
        img_reconstruida = outputs.detach().cpu().numpy()[0]
        plt.figure()
        plt.axis("off")
        a=img_reconstruida.transpose(1, 2, 0)
        a[a < 0] = 0
        a[a > 1] = 1
        plt.imshow(a)
        plt.savefig(f'imagenes_reconstruidas_train/img_reconstruida_{img_count}.png')
        plt.close() 
        img_count += 1

    #np.array(img,np.int32)
    #print(np.max(img_reconstruida.transpose(1, 2, 0)))
    #print(np.min(img_reconstruida.transpose(1, 2, 0)))
    #print((img_reconstruida.transpose(1, 2, 0)*255).astype('uint8'))
    loss = loss / len(loader)
    print(f"Train loss = {loss:.6f}")
    losses.append(loss)
       
    return loss    


def test(model, loader, optimizer, criterion, device):
    loss = 0
    model.eval()

        
    for batch_features in loader:
    
        batch_features = batch_features.float() / 255.0
        batch_features = batch_features.permute(0, 3, 1, 2) 

        batch_features = batch_features.to(device)

        with torch.no_grad():
            outputs = model(batch_features)
            
        
        outputs = F.interpolate(outputs, size=DIM, mode='bilinear', align_corners=False)
        
        test_loss = criterion(outputs, batch_features)
        
        loss += test_loss.item()


    loss = loss / len(loader)

    return loss



def test_print(model, loader, optimizer, criterion, device):
    loss = 0
    model.eval()
    img_count=0
        
    for batch_features in tqdm(loader):
    
        batch_features = batch_features.float() / 255.0
        batch_features = batch_features.permute(0, 3, 1, 2) 

        batch_features = batch_features.to(device)

        with torch.no_grad():
            outputs = model(batch_features)
            
        
        outputs = F.interpolate(outputs, size=DIM, mode='bilinear', align_corners=False)
        
        test_loss = criterion(outputs, batch_features)
        
        loss += test_loss.item()

        #print("Imatge original")
        img_original = batch_features.detach().cpu().numpy()
        plt.close() 
        plt.figure()
        plt.axis("off")
        
        plt.imshow(img_original[0].transpose(1, 2, 0))
        plt.savefig(f'imagenes_originales_test/img_original_{img_count}.png')
        plt.close() 
       
        #print("Imatge reconstruida")
        
        img_reconstruida = outputs.detach().cpu().numpy()[0]
        plt.figure()
        plt.axis("off")
        a=img_reconstruida.transpose(1, 2, 0)
        a[a < 0] = 0
        a[a > 1] = 1
        plt.imshow(a)
        plt.savefig(f'imagenes_reconstruidas_test/img_reconstruida_{img_count}.png')
        plt.close() 
        img_count += 1


    loss = loss / len(loader)

    return loss


print("uso de modelo")
model = ConvAE().to(device)

optimizer = torch.optim.Adamax(model.parameters(),  lr=0.001 )

criterion = nn.MSELoss()

losses=[]
losses_test=[]
print("epocas")
epochs = 350
#he canviado esto

for epoch in tqdm(range(epochs-1)):


    loss=train(model, loader_train, optimizer, criterion, device)
    print("no ultima")
    losses.append(loss)
    
    #if (epoch%2)==0:
    loss_test=test(model, loader_test, optimizer, criterion, device)
    losses_test.append(loss_test)

#este guarda las imagenes
loss=train_print(model, loader_train, optimizer, criterion, device)
losses.append(loss) 
       
loss_test=test_print(model, loader_test, optimizer, criterion, device)
losses_test.append(loss_test)        

  
  
os.makedirs('modelos', exist_ok=True) 
nom=str(losses[-1])

torch.save(model, "modelos/modelos_"+nom+'_modelo50')
data=(losses,losses_test)

file = open('loses_pickle_50', 'wb')
# dump information to that file
pickle.dump(data, file)
# close the file
file.close()


#plt.close()
plt.figure()
#plt.plot(np.log(losses[1:]))
#plt.imshow(img_reconstruida[0].transpose(1, 2, 0))
plt.plot((losses[5:]))
plt.savefig('loss_train.png')
#plt.close()
plt.figure()
plt.plot(losses_test[:])
plt.savefig('loss_test.png')
plt.close()
#import pandas as pd
from matplotlib import pyplot as plt
#import cv2 as cv
import numpy as np
import os
import pickle
from sklearn.metrics import roc_curve,accuracy_score,f1_score


file_10 = open('10/rgbymodelo10', 'rb')
file_20 = open('20/rgbymodelo20', 'rb')
file_30 = open('30/rgbymodelo30', 'rb')
file_50 = open('50/rgbymodelo50', 'rb')


data_10 = pickle.load(file_10)
data_20 = pickle.load(file_20)
data_30 = pickle.load(file_30)
data_50 = pickle.load(file_50)

# close the file
file_10.close()
file_20.close()
file_30.close()
file_50.close()


_ , _ , l_blaves_10 , seva_y_10 = np.array(data_10)[:,:500]
_ , _ , l_blaves_20 , seva_y_20 = np.array(data_20)[:,:500]
_ , _ , l_blaves_30 , seva_y_30 = np.array(data_30)[:,:500]
_ , _ , l_blaves_50 , seva_y_50 = np.array(data_50)[:,:500]

_ , _ , t_blaves_10 , t_y_10 = np.array(data_10)[:,500:]
_ , _ , t_blaves_20 , t_y_20 = np.array(data_20)[:,500:]
_ , _ , t_blaves_30 , t_y_30 = np.array(data_30)[:,500:]
_ , _ , t_blaves_50 , t_y_50 = np.array(data_50)[:,500:]

#10
predicted=np.array(l_blaves_10)/np.max(l_blaves_10)
target=np.array(seva_y_10)
fpr, tpr, threshold = roc_curve(target, predicted)

esquina=np.array((0,1))
dist_min=100000
mes_proper=-1
for x,y,t in zip(fpr,tpr,threshold):
    dist= np.linalg.norm(np.array((x,y)) - esquina)
    if dist<dist_min:
        dist_min=dist
        mes_poper=x,y,t

plt.figure()
plt.plot(fpr, tpr,
    color="darkorange",
    label="ROC curve"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.axvline(x = mes_poper[0], color = 'g', label = 'treshold', linestyle=':')
plt.axhline(y = mes_poper[1], color = 'g', label = 'treshold', linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc blau 10")
plt.legend(loc="lower right")
plt.show()

p=predicted>mes_poper[2]
v=target>0

print(f1_score(v,p),mes_poper[1],mes_poper[0],accuracy_score(v,p))





#20
predicted=np.array(l_blaves_20)/np.max(l_blaves_20)
target=np.array(seva_y_20)
fpr, tpr, threshold = roc_curve(target, predicted)

esquina=np.array((0,1))
dist_min=100000
mes_proper=-1
for x,y,t in zip(fpr,tpr,threshold):
    dist= np.linalg.norm(np.array((x,y)) - esquina)
    if dist<dist_min:
        dist_min=dist
        mes_poper=x,y,t

plt.figure()
plt.plot(fpr, tpr,
    color="darkorange",
    label="ROC curve"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.axvline(x = mes_poper[0], color = 'g', label = 'treshold', linestyle=':')
plt.axhline(y = mes_poper[1], color = 'g', label = 'treshold', linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc blau 20")
plt.legend(loc="lower right")
plt.show()

p=predicted>mes_poper[2]
v=target>0

print(f1_score(v,p),mes_poper[1],mes_poper[0],accuracy_score(v,p))



#30
predicted=np.array(l_blaves_30)/np.max(l_blaves_30)
target=np.array(seva_y_30)
fpr, tpr, threshold = roc_curve(target, predicted)

esquina=np.array((0,1))
dist_min=100000
mes_proper=-1
for x,y,t in zip(fpr,tpr,threshold):
    dist= np.linalg.norm(np.array((x,y)) - esquina)
    if dist<dist_min:
        dist_min=dist
        mes_poper=x,y,t


plt.figure()
plt.plot(fpr, tpr,
    color="darkorange",
    label="ROC curve"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.axvline(x = mes_poper[0], color = 'g', label = 'treshold', linestyle=':')
plt.axhline(y = mes_poper[1], color = 'g', label = 'treshold', linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc blau 30")
plt.legend(loc="lower right")
plt.show()
p=predicted>mes_poper[2]
v=target>0

print(f1_score(v,p),mes_poper[1],mes_poper[0],accuracy_score(v,p))


#50
predicted=np.array(l_blaves_50)/np.max(l_blaves_50)
target=np.array(seva_y_50)
fpr, tpr, threshold = roc_curve(target, predicted)

esquina=np.array((0,1))
dist_min=100000
mes_proper=-1
for x,y,t in zip(fpr,tpr,threshold):
    dist= np.linalg.norm(np.array((x,y)) - esquina)
    if dist<dist_min:
        dist_min=dist
        mes_poper=x,y,t

plt.figure()
plt.plot(fpr, tpr,
    color="darkorange",
    label="ROC curve"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.axvline(x = mes_poper[0], color = 'g', label = 'treshold', linestyle=':')
plt.axhline(y = mes_poper[1], color = 'g', label = 'treshold', linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Roc blau 50")
plt.legend(loc="lower right")
plt.show()

p=predicted>mes_poper[2]
v=target>0

print(f1_score(v,p),mes_poper[1],mes_poper[0],accuracy_score(v,p))


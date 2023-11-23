#import pandas as pd
from matplotlib import pyplot as plt
#import cv2 as cv
import numpy as np
import os
import pickle



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


l_vermelles_10 , loses_10 , l_blaves_10 , seva_y_10 = np.array(data_10)[:,:500]
l_vermelles_20 , loses_20 , l_blaves_20 , seva_y_20 = np.array(data_20)[:,:500]
l_vermelles_30 , loses_30 , l_blaves_30 , seva_y_30 = np.array(data_30)[:,:500]
l_vermelles_50 , loses_50 , l_blaves_50 , seva_y_50 = np.array(data_50)[:,:500]


#10
x_r = np.array(l_vermelles_10)
x_g = np.array(loses_10)
x_b = np.array(l_blaves_10)
y = np.array(seva_y_10)
m = np.ma.masked_where(y>0, y)#1
xr_1= x_r[m.mask]
xg_1= x_g[m.mask]
xb_1= x_b[m.mask]
m = np.ma.masked_where(y<0, y)#-1
xr_2= x_r[m.mask]
xg_2= x_g[m.mask]
xb_2= x_b[m.mask]

plt.title("historograma positius vermell 10")
plt.hist(xr_1, bins=100,range=[0,70])#positius 1
plt.savefig("historograma positius vermell 10")
plt.show()
plt.title("historograma negatius vermell 10")
plt.hist(xr_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("historograma negatius vermell 10")
plt.show()

plt.title("historograma positius general 10")
plt.hist(xg_1, bins=100,range=[0,70])#positius 1
plt.savefig("historograma positius general 10")
plt.show()
plt.title("historograma negatius general 10")
plt.hist(xg_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("historograma negatius general 10")
plt.show()

plt.title("historograma positius blau 10")
plt.hist(xb_1, bins=100,range=[0,70])#positius 1
plt.savefig("historograma positius blau 10")
plt.show()
plt.title("historograma negatius blau 10")
plt.hist(xb_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("historograma negatius blau 10")
plt.show()

#20

x_r = np.array(l_vermelles_20)
x_g = np.array(loses_20)
x_b = np.array(l_blaves_20)
y = np.array(seva_y_20)
m = np.ma.masked_where(y>0, y)#1
xr_1= x_r[m.mask]
xg_1= x_g[m.mask]
xb_1= x_b[m.mask]
m = np.ma.masked_where(y<0, y)#-1
xr_2= x_r[m.mask]
xg_2= x_g[m.mask]
xb_2= x_b[m.mask]

plt.title("historograma positius vermell 20")
plt.hist(xr_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius vermell 10")
plt.show()
plt.title("historograma negatius vermell 20")
plt.hist(xr_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius vermell 20")
plt.show()

plt.title("historograma positius general 20")
plt.hist(xg_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius general 20")
plt.show()
plt.title("historograma negatius general 20")
plt.hist(xg_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius general 20")
plt.show()

plt.title("historograma positius blau 20")
plt.hist(xb_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius blau 20")
plt.show()
plt.title("historograma negatius blau 20")
plt.hist(xb_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius blau 20")
plt.show()


#30

x_r = np.array(l_vermelles_30)
x_g = np.array(loses_30)
x_b = np.array(l_blaves_30)
y = np.array(seva_y_30)
m = np.ma.masked_where(y>0, y)#1
xr_1= x_r[m.mask]
xg_1= x_g[m.mask]
xb_1= x_b[m.mask]
m = np.ma.masked_where(y<0, y)#-1
xr_2= x_r[m.mask]
xg_2= x_g[m.mask]
xb_2= x_b[m.mask]


plt.title("historograma positius vermell 30")
plt.hist(xr_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius vermell 30")
plt.show()
plt.title("historograma negatius vermell 30")
plt.hist(xr_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius vermell 30")
plt.show()

plt.title("historograma positius general 30")
plt.hist(xg_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius general 30")
plt.show()
plt.title("historograma negatius general 30")
plt.hist(xg_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius general 30")
plt.show()

plt.title("historograma positius blau 30")
plt.hist(xb_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius blau 30")
plt.show()
plt.title("historograma negatius blau 30")
plt.hist(xb_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius blau 30")
plt.show()

#50

x_r = np.array(l_vermelles_50)
x_g = np.array(loses_50)
x_b = np.array(l_blaves_50)
y = np.array(seva_y_50)
m = np.ma.masked_where(y>0, y)#1
xr_1= x_r[m.mask]
xg_1= x_g[m.mask]
xb_1= x_b[m.mask]
m = np.ma.masked_where(y<0, y)#-1
xr_2= x_r[m.mask]
xg_2= x_g[m.mask]
xb_2= x_b[m.mask]

plt.title("historograma positius vermell 50")
plt.hist(xr_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius vermell 50")
plt.show()
plt.title("historograma negatius vermell 50")
plt.hist(xr_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius vermell 50")
plt.show()

plt.title("historograma positius general 50")
plt.hist(xg_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius general 50")
plt.show()
plt.title("historograma negatius general 50")
plt.hist(xg_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius general 50")
plt.show()

plt.title("historograma positius blau 50")
plt.hist(xb_1, bins=100,range=[0,70])#positius 1
plt.savefig("plots/historograma positius blau 50")
plt.show()
plt.title("historograma negatius blau 50")
plt.hist(xb_2, bins=100,range=[0,70])#negatius -1 
plt.savefig("plots/historograma negatius blau 50")
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:12:22 2023

@author: Yasmin
"""

import pickle 
import matplotlib.pyplot as plt 
import numpy as np

file = open('C:/Users/Yamin/Downloads/imagenes psiv/pickle50/loses_pickle_50', 'rb')


data = pickle.load(file)

#0 losses
#1 losses test 

#descomentar una u otra dependiendo si es de train o test la 
#loss que queremos visualizar 

# plt.plot(data[0][1:], label='Loss train')

plt.plot(data[1], label='Loss test')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gràfica Loss test 50 epoques i 350 epoques')


plt.legend()
plt.show()

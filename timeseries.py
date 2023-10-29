# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:59:32 2021

@author: JunJi

special parameter for 1-110 model
"""
from scipy.signal import find_peaks, peak_widths 
from scipy.integrate import odeint
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

############################ two node enemy dynamics model############################ 
###############S-->A  B:output   A, B has production and degradation term#############
#reference:2018-cellsystem-Incoherent Inputs Enhance the Robustness of Biological Oscillators
def motif_odefun(Protein, t):  
    
    A, B = Protein
    
    #[kS, k1, k2, k3, k4, j1, j2, j3, j4, kinhA, kinhB, n1, n2, n3, n4] =  para
    
    # for A
    if(couple_matrix[0]==-1.0):
        term_1 = couple_matrix[0]*k1*A*A**n1/(A**n1+j1**n1)
    elif(couple_matrix[0]==1.0):
        term_1 = couple_matrix[0]*k1*A*(1-A)**n1/((1-A)**n1+j1**n1)
    else:
        term_1 = 0.0
        
    if(couple_matrix[1]==-1.0):
        term_2 = couple_matrix[1]*k2*B*A**n2/(A**n2+j2**n2)        
    elif(couple_matrix[1]==1.0):
        term_2 = couple_matrix[1]*k2*B*(1-A)**n2/((1-A)**n2+j2**n2)
    else:
        term_2 = 0.0

    # for B
    if(couple_matrix[2]==-1.0):
        term_3 = couple_matrix[2]*k3*A*B**n3/(B**n3+j3**n3)
    elif(couple_matrix[2]==1.0):
        term_3 = couple_matrix[2]*k3*A*(1-B)**n3/((1-B)**n3+j3**n3)        
    else:
        term_3 = 0.0
    
    if(couple_matrix[3]==-1.0):
        term_4 = couple_matrix[3]*k4*B*B**n4/(B**n4+j4**n4)
    elif(couple_matrix[3]==1.0):
        term_4 = couple_matrix[3]*k4*B*(1-B)**n4/((1-B)**n4+j4**n4)        
    else:
        term_4 = 0.0


    term_SA = kS*(1-A)    
                           
    term_inhA = kinhA*A
    term_inhB = kinhB*B

    dA = term_SA  + term_1 + term_2 - term_inhA
    dB = term_3 + term_4 - term_inhB
    
    return np.array([dA, dB])
#####################################################################################
    

couple_matrix = [ 1., -1., 1., 0.]   #Motif1-11-1

protein_init = [0.0, 0.0]  #initial value

filename = './v1_v5_H_amplitude_peak_biphase_para.csv'
f1=open(filename,"rb")
case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
f1.close()

case_train = np.array(case_train) 

k = 53

for kS in np.arange(0.1, 0.2, 0.2):
    print(kS)
    
    for j in range(k, k+1): 
   
        t_end = 100
        dt = 0.1
        t = np.arange(0, t_end, dt)
        m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
        
        para = case_train[j, 3:]
           
        #[kS, k1, k2, k3, k4, j1, j2, j3, j4, kinhA, kinhB, n1, n2, n3, n4] =  para
        
        kS = 0.4
#        k1 = para[1] 
#        k2 = para[2]
#        k3 = para[3] 
#        k4 = para[4]
#        j1 = para[5] 
#        j2 = para[6] 
#        j3 = para[7] 
#        j4 = para[8]
#        kinhA = para[9] 
#        kinhB = para[10] 
        
        k1 = round(para[1], 2)  # 
        k2 = round(para[2], 2)  #
        k3 = round(para[3], 2)  # 
        k4 = round(para[4], 2)  #
        j1 = round(para[5], 2)  # 
        j2 = round(para[6], 4)  #
        j3 = round(para[7], 2)  #
        j4 = round(para[8], 2)  #
        kinhA = round(para[9], 3)  #
        kinhB = round(para[10], 2)  #           
        
        n1 = para[11] 
        n2 = para[12] 
        n3 = para[13] 
        n4 = para[14]
           
        track = odeint(motif_odefun, protein_init, t)
        
        NN = track[-m:-1,1]
        
        max(NN) < 1e-6 or max(NN) - min(NN) < 1e-6*max(NN)
        print(max(NN), min(NN))
        

    plt.figure(1)
    plt.plot(t, track[:,0], label='A')#[:,0]
    plt.plot(t, track[:,1], label='B')
    plt.ylabel('B')
    #plt.xlim(0, 20)
    #plt.ylim(0, 6)
    plt.legend()
    
#    plt.figure(2)
#    plt.plot(track[69000:,0], track[69000:,1], label='ks='+format(kS, '.1f'))
#    plt.legend()   
#   
#M= 69000   
#    
#flow = np.zeros((1000, 2)) 
#for j in range(len(flow)):
#    
#    track[M, 0]
#    
#    flow[j][0] = kS*(1-track[j+M, 0]) + k1*track[j+M, 0]*(1-track[j+M, 0])**n1/((1-track[j+M, 0])**n1+j1**n1) - k2*track[j+M, 1]*track[j+M, 0]**n2/(track[j+M, 0]**n2+j2**n2) - kinhA*track[j+M, 0]
#    
#    flow[j][1] = k3*track[j+M, 0]*(1-track[j+M, 1])**n3/((1-track[j+M, 1])**n3+j3**n3) - kinhB*track[j+M, 1]   
#   

#plt.figure(3)
#plt.plot(flow[:, 0], flow[:, 1], label='ks='+format(kS, '.1f'))
#plt.legend()  
#plt.savefig('ks_0.1_0.3_0.5_0.7_timeseries.jpg', dpi=300)

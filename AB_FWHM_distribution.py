# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:33:33 2020

@author: JunJi
"""
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import odeint
import numpy as np
from pylab import array, linspace, subplots
import pandas as pd
from sklearn import preprocessing
from itertools import product

      
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


protein_init = [0.0, 0.0]  #initial value

couple_matrix = [ 1., -1., 1., 0.]   #Motif1-11-1couple_matrix = [ 1., -1.,  1., -1.]   #Motif1-11-1     
    
    
#f1=open('./LHS'+str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3]))+'.csv', "rb")

filename = './v1_Motif' + str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3])) + 'oscill_para.csv'
f1=open(filename, "rb")

case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
f1.close()    
LHS_of_paras=np.array(case_train)

N = len(LHS_of_paras)

ABFWHM_distr = []

#k = 36   
for j in range(1000):    
#for j in range(k, k+1):
    print([j, 1000])
    
    t_end = 1000
    dt = 0.001
    t = np.arange(0, t_end, dt)
    m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
    
    para = LHS_of_paras[j, 3:]    #运行LHS文件时要将3改为0
    
    #[kS, k1, k2, k3, k4, j1, j2, j3, j4, kinhA, kinhB, n1, n2, n3, n4] =  para
    
    kS = para[0] 
    k1 = para[1] 
    k2 = para[2]
    k3 = para[3] 
    k4 = para[4]
    j1 = para[5] 
    j2 = para[6] 
    j3 = para[7] 
    j4 = para[8]
    kinhA = para[9] 
    kinhB = para[10] 
    n1 = para[11] 
    n2 = para[12] 
    n3 = para[13] 
    n4 = para[14]   
    
    track = odeint(motif_odefun, protein_init, t)
    
    NNA = track[-m:-1, 0]
    NNB = track[-m:-1, 1]
    
    valleysA, _ = find_peaks(-1*NNA, prominence=0.01)  #prominence峰的突出程度  
    valleysB, _ = find_peaks(-1*NNB, prominence=0.01)  #prominence峰的突出程度  
    
    if(len(valleysA)>6 and len(valleysB)>6):
  
        last5periodA = NNA[valleysA[-6]:valleysA[-1]]# choose the timeseries in last 5 period
        period = (valleysB[-1]-valleysB[-6])*dt/5
        peaks5A, _ = find_peaks(last5periodA, prominence=0.01*max(last5periodA))#-m:-1
        results_halfA = peak_widths(last5periodA, peaks5A, rel_height=0.5)    #半高宽                
        FWHMA = sum(results_halfA[0]*dt)/len(results_halfA[0])                 
        
        #print(FWHMA/period)  
    
    
        last5periodB = NNB[valleysB[-6]:valleysB[-1]]# choose the timeseries in last 5 period
        #period = (valleysB[-1]-valleysB[-6])*dt/5
        peaks5B, _ = find_peaks(last5periodB, prominence=0.01*max(last5periodB))#-m:-1
        results_halfB = peak_widths(last5periodB, peaks5B, rel_height=0.5)    #半高宽                
        FWHMB = sum(results_halfB[0]*dt)/len(results_halfB[0])                 
        
        #print(FWHMB/period)  
           
        ABFWHM_distr.append([FWHMA/period, FWHMB/period])
    
#np.savetxt('./ABFWHM_distr.csv', ABFWHM_distr, delimiter = ',') #保存参数

    
    
    plt.ylim([0, 1])
    plt.figure(j)
    plt.plot(t, track[:,0], label='A')#[:,1]
    plt.plot(t, track[:,1], label='B')#[:,1]
    plt.legend()
#
     

################calculate if every FWHMA/period bigger than FWHMB/period################

#filename = './ABFWHM_distr.csv'
#f1=open(filename, "rb")
#
#case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
#f1.close()    
#ABFWHM_distr=np.array(case_train)
#
#Num = len(ABFWHM_distr)
#num = 0
#
##k = 4   
#for j in range(Num): 
#    if(ABFWHM_distr[j, 0]>ABFWHM_distr[j, 1]):
#        num = num + 1
#        print(j)
#
#print(num)
########################################################################################  
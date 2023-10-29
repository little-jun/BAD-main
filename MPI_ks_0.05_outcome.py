# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 08:52:20 2021

@author: JunJi
"""
from mpi4py import MPI
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import odeint
import numpy as np
#import pandas as pd
from itertools import product
import time

t0 = time.time() 
     
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


###############judge if the timeseries is steady oscillation ##################
def judge_oscill(NN_A, NN_B, t_end):
    '''
    NN_B #输入，为一个列表或者数组
    输出：0表示不震荡, 1表示稳定震荡       #2表示需要更长时间进行再次判断
    '''
    dt = 0.001
    
    if(max(NN_B) < 1e-6 or max(NN_B) - min(NN_B) < 1e-6*max(NN_B)):   #判定为稳态
        return 0
    
    else:
        peaks_A, _ = find_peaks(NN_A, prominence=0.01)  #-m:-1  #use absolute value of amplitude to judge oscillator
        valleys_A, _ = find_peaks(-1*NN_A, prominence=0.01)  #prominence峰的突出程度  
        
        peaks, _ = find_peaks(NN_B, prominence=0.01)  #-m:-1  #use absolute value of amplitude to judge oscillator
        valleys, _ = find_peaks(-1*NN_B, prominence=0.01)  #prominence峰的突出程度  

        if(len(peaks)>5 and len(valleys)>5 and len(peaks_A)>5 and len(valleys_A)>5 ):   #判定为稳定震荡 and max(NN_B) - min(NN_B) > 0.01*max(NN_B)  and np.abs(NN_B[peaks[-1]]-NN_B[peaks[-6]])<1e-3 and np.abs(NN_B[valleys[-1]]-NN_B[valleys[-6]])<1e-3
           
            last5_peak_value = [NN_B[peaks[x]] for x in range(-5,0)]
            last5_period = [(valleys[x]-valleys[x-1])*dt for x in range(-5,0)]
            
            if(np.std(last5_peak_value)/np.mean(last5_peak_value)<0.01 and np.std(last5_period)/np.mean(last5_period)<0.01):#判定为稳定震荡

                A_last5period = NN_A[valleys[-6]:valleys[-1]]# choose the timeseries of A in last 5 period
                max_A = max(A_last5period)  # Find the maximum y value
                min_A = min(A_last5period)  # Find the minmum y value
                amplitude_A = max_A - min_A  
                
                last5period = NN_B[valleys[-6]:valleys[-1]]# choose the timeseries in last 5 period
                max_y = max(last5period)  # Find the maximum y value
                min_y = min(last5period)  # Find the minmum y value
                amplitude = max_y - min_y
                period = (valleys[-1]-valleys[-6])*dt/5
#                peaks5, _ = find_peaks(last5period, prominence=0.01*max(last5period))#-m:-1
#                results_half = peak_widths(last5period, peaks5, rel_height=0.5)    #半高宽                
#                FWHM = sum(results_half[0]*dt)/len(results_half[0])  
                
                if(peaks_A[0] < peaks[0]): #B峰的位置在A的右边
                    phase_difference = (peaks[0] - peaks_A[0])*dt
                else:
                    phase_difference = period - (peaks_A[0] - peaks[0])*dt
                    
                if(peaks_A[0] < valleys_A[0]): #A峰的位置在A谷的左边
                    decay_phase_A = (valleys_A[0] - peaks_A[0])*dt  #A的衰减相
                else:
                    decay_phase_A = period - (peaks_A[0] - valleys_A[0])*dt
            
                rise_phase_A = period - decay_phase_A   #A的增加相    
               
                if(peaks[0] < valleys[0]): #B峰的位置在B谷的左边
                    decay_phase_B = (valleys[0] - peaks[0])*dt  #B的衰减相
                else:
                    decay_phase_B = period - (peaks[0] - valleys[0])*dt
            
                rise_phase_B = period - decay_phase_B   #B的增加相
                
                return max_A, min_A, amplitude_A, max_y, min_y, amplitude, period, phase_difference, rise_phase_A, decay_phase_A, rise_phase_B, decay_phase_B
            
            else:  #需要更长时间进行再次判断
            
                #return 2
                while (t_end < 5000):   #最大运行时间为 50000
                   
                    t_end = t_end + 2300
                    dt = 0.001
                    t = np.arange(0, t_end, dt)
                    m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
                    
                    track = odeint(motif_odefun, protein_init, t)
                    
                    NN_A = track[-m:-1,0]
                    NN_B = track[-m:-1,1]
                    
                    return judge_oscill(NN_A, NN_B, t_end)
                
                return     #放弃这组参数
        
        else:  #需要更长时间进行再次判断
            
            #return 2
            while (t_end < 5000):   #最大运行时间为 50000
               
                t_end = t_end + 2300
                dt = 0.001
                t = np.arange(0, t_end, dt)
                m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
                
                track = odeint(motif_odefun, protein_init, t)
                
                NN_A = track[-m:-1,0]
                NN_B = track[-m:-1,1]
                
                return judge_oscill(NN_A, NN_B, t_end)
            
            return     #放弃这组参数
###############################################################################

couple_matrix = [ 1., -1., 1., 0.]   #Motif1-11-1


filename = './v1_v5_H_amplitude_peak_biphase_para.csv'
f1=open(filename,"rb")
case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
f1.close()

case_train = np.array(case_train[:, 3:])  #Remove the first three columns(period, amplitude, FWHM/period) from the csv file, get parameter


N = len(case_train)   #50000#    #取多少组参数

#################################并行代码#################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
nprocs = comm.Get_size() 

nsteps = N   ####比较重要


if rank == 0: # proc0
    # determine the size of each sub-task
    ave, res = divmod(nsteps, nprocs)    
    counts = [ave + 1 if p < res else ave for p in range(nprocs)] # e.g. counts = [ave + 1, ave + 1, ave + 1, ave + 1, ave , ave , ave ,...] (res =4)

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    # save the starting and ending indices in data
    data = [(starts[p], ends[p]) for p in range(nprocs)] # e.g. data = [(0,10),(10,30),(30,100)]

else:
    data = None

data = comm.scatter(data, root=0) # root=0 represents scatter from proc0
###########################################################################


protein_init = [0.0, 0.0]  #initial value
        
filename = 'ks_0.05_outcome.csv'

f=open(filename, 'a')
#rows: response to different parameters(only oscillators are shown)
#columns: from left to right: period, amplitude, FWHM/period, parameters    
    
for j in range(data[0], data[1]): 
#for j in range(N):     
    
    #if(j%10000==0):
    print([j, N])
    
    t_end = 200
    dt = 0.001
    t = np.arange(0, t_end, dt)
    m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡

    
    para = case_train[j, :]
    
    
    #[kS, k1, k2, k3, k4, j1, j2, j3, j4, kinhA, kinhB, n1, n2, n3, n4] =  para
    
    kS = 0.05#para[0] 
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
    
    
    if(np.max(track) <= 1 and np.min(track) >= 0):  #初步解序列有没有越界
        
        NN_A = track[-m:-1,0]
        NN_B = track[-m:-1,1]   

        value_oscill = judge_oscill(NN_A, NN_B, t_end)
    
        if(type(value_oscill) == tuple):
            #oscill_para.append(all_Para)
            f.write(str(list(value_oscill) + list(para)).replace("[","").replace("]","") + "\n")
        else:
            t = np.arange(0, 3000, 0.01)
            m = int(0.5*t_end/dt) 
            track = odeint(motif_odefun, protein_init, t)
            NN_B = track[-m:-1,1]                                                                                                       
            f.write(str([max(NN_A), min(NN_A), max(NN_A)-min(NN_A), max(NN_B), min(NN_B), max(NN_B)-min(NN_B), 0, 0, 0, 0, 0, 0] + list(para)).replace("[","").replace("]","") + "\n")

            #max_y, min_y, amplitude, period,phase_difference, rise_phase_B, decay_phase_B


f.close() 

MPI.Finalize() 
print(time.time() - t0)   
print('The code have run')
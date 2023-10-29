# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:33:33 2020

@author: JunJi
"""
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
import time

t0 = time.time()

count = 0
count1 = 0

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
def judge_oscill(NN, t_end):
    '''
    NN #输入，为一个列表或者数组
    输出：0表示不震荡, 1表示稳定震荡       #2表示需要更长时间进行再次判断
    '''
    dt = 0.001
    
    if(max(NN) < 1e-6 or max(NN) - min(NN) < 1e-6*max(NN)):   #判定为稳态
        return 0
    
    else:
    
        peaks, _ = find_peaks(NN, prominence=0.01)  #-m:-1  #use absolute value of amplitude to judge oscillator
        valleys, _ = find_peaks(-1*NN, prominence=0.01)  #prominence峰的突出程度  

        if(len(peaks)>5 and len(valleys)>5):   #判定为稳定震荡 and max(NN) - min(NN) > 0.01*max(NN)  and np.abs(NN[peaks[-1]]-NN[peaks[-6]])<1e-3 and np.abs(NN[valleys[-1]]-NN[valleys[-6]])<1e-3
           
            last5_peak_value = [NN[peaks[x]] for x in range(-5,0)]
            last5_period = [(valleys[x]-valleys[x-1])*dt for x in range(-5,0)]
            
            if(np.std(last5_peak_value)/np.mean(last5_peak_value)<0.01 and np.std(last5_period)/np.mean(last5_period)<0.01):#判定为稳定震荡

                last5period = NN[valleys[-6]:valleys[-1]]# choose the timeseries in last 5 period
                max_y = max(last5period)  # Find the maximum y value
                min_y = min(last5period)  # Find the minmum y value
                amplitude = max_y - min_y
                period = (valleys[-1]-valleys[-6])*dt/5
                peaks5, _ = find_peaks(last5period, prominence=0.01*max(last5period))#-m:-1
                results_half = peak_widths(last5period, peaks5, rel_height=0.5)    #半高宽                
                FWHM = sum(results_half[0]*dt)/len(results_half[0])                 
                
                return period, amplitude, FWHM/period
            
            else:  #需要更长时间进行再次判断
            
                #return 2
                while (t_end < 5000):   #最大运行时间为 50000
                   
                    t_end = t_end + 2300
                    dt = 0.001
                    t = np.arange(0, t_end, dt)
                    m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
                    
                    track = odeint(motif_odefun, protein_init, t)
                    
                    NN = track[-m:-1,1]
                    
                    return judge_oscill(NN, t_end)
                
                return     #放弃这组参数
        
        else:  #需要更长时间进行再次判断
            
            #return 2
            while (t_end < 5000):   #最大运行时间为 50000
               
                t_end = t_end + 2300
                dt = 0.001
                t = np.arange(0, t_end, dt)
                m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
                
                track = odeint(motif_odefun, protein_init, t)
                
                NN = track[-m:-1,1]
                
                return judge_oscill(NN, t_end)
            
            return     #放弃这组参数
###############################################################################


# motif=np.array([[ 1., -1., 1., -1.], [ 1., -1., 1., 0.], [ 1., -1., 1., 1.]])   
            
couple_matrix = [1., -1., 1., 0.]   #Motif1-11-1

protein_init = [0.0, 0.0]  #initial value
 
filename = './Motif' + str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3])) + 'oscill_para.csv'
f1=open(filename,"rb")
case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
f1.close()

case_train = np.array(case_train[:, 3:])  #Remove the first three columns(period, amplitude, FWHM/period) from the csv file, get parameter


#################################并行代码#################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
nprocs = comm.Get_size() 

nsteps = len(case_train)   ####比较重要


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


para_outcome = []

filename = './Motif' + str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3])) + 'biphase_para.csv'
f=open(filename, 'a')
#rows: response to different parameters(only oscillators are shown)
#columns: from left to right: type of biphase of (period, amplitude, FWHM/period),  and parameters 
# 0: no biphase   1:peak type    2:valley type    3:mix type



#for j in range(len(case_train)):
#    print([j, len(case_train)])
for j in range(data[0], data[1]):  
    print([j, len(case_train)])  
    
    abnormal = 0 # 对于每一组参数没有出现异常
    
    t_end = 200
    dt = 0.001 
    t = np.arange(0, t_end, dt)
    m = int(0.5*t_end/dt)   #在最后m步通过比较最大值，最小值判断是否震荡
    
    period_amplitude_FWHM = [[0, 0, 0]]

    para = list(case_train[j,:]) #
       
#   kS = para[0] 
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
    
    bifr_para = np.arange(0, 1.05, 0.05)  #hopf分支参数bifr_para_start, bifr_para_end, bifr_para_step
    
    
    for kS in bifr_para:
        #print(kS)

        track = odeint(motif_odefun, protein_init, t)
    
        if(np.max(track) <= 1 and np.min(track) >= 0):  #初步解序列有没有越界
            
            NN = track[-m:-1,1]   
    
            value_oscill = judge_oscill(NN, t_end)
        
#            if(value_oscill == 0.0): #稳态
#                period_amplitude_FWHM.append([0, 0, 0])
#                
#            elif(type(value_oscill) == tuple):                              
#                period_amplitude_FWHM.append(list(value_oscill)) #period_amplitude_FWHM 
#            
#            else:
#                abnormal = 1   #在给定最大时间，依然不能判断出事稳态还是震荡，故舍弃这组参数
#                count = count + 1
#                break
                            
            if(type(value_oscill) == tuple):                              
                period_amplitude_FWHM.append(list(value_oscill)) #period_amplitude_FWHM 
            
            else:
                period_amplitude_FWHM.append([0, 0, 0]) 

        else:
            abnormal = 1     #解序列越界, 舍弃这组参数
            count1 = count1 + 1
            break
        
    if(abnormal == 1.0):
        continue

    #判断在震荡参数下，改变其中一个参数，对period_amplitude_FWHM 是否以及双相调控的形式   
    period_amplitude_FWHM = np.array(period_amplitude_FWHM)  
    
    
    biphase_type = [0, 0, 0]

    for i in range(3):
        PAF = list(period_amplitude_FWHM[:,i])
        PAF_POSindex = [x for x in range(len(PAF)) if PAF[x] > 0.0]
    
        if(len(PAF_POSindex) > 0.0):
            posPAF = PAF[min(PAF_POSindex):max(PAF_POSindex)+1]
            
            inverse_posPAF=[-l for l in posPAF]
            
            peaks1, _ = find_peaks(posPAF, prominence=0.01*max(posPAF))  #峰值必须大于0.005，才认为是双相调控
            valley1, _ = find_peaks(inverse_posPAF, prominence=0.01*max(posPAF))
             
            if(len(peaks1)+len(valley1)>0):     #判断振幅变化产生双相调控的类型 #周期加+1
                if(len(peaks1)>=1 and len(valley1)==0): #峰形双相调控
                    #biphase_type.append(1)
                    biphase_type[i]=1
                elif(len(peaks1)==0 and len(valley1)>=1): #谷形双相调控
                    biphase_type[i]=2            
                elif(len(peaks1)>=1 and len(valley1)>=1):  #mix形双相调控
                    biphase_type[i]=3 
#            else:
                #biphase_type.append(0)
        else:
            break
        # plt.figure(i)
        # plt.plot(posPAF)
          
    f.write(str(biphase_type + para).replace("[","").replace("]","") + "\n")
#    para_outcome.append(biphase_type + para)
        
f.close() 
MPI.Finalize()     
print(count1)  
print(time.time() - t0)

#np.savetxt('test_find_biphase_para.csv', para_outcome, delimiter = ',') #保存参数, fmt='%.5f'  
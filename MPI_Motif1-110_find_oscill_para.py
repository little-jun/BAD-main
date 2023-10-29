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


##################################LHS参数抽样###################################
def LHSample( D,bounds,N):#直接输出抽样  
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''

    result = np.empty([N, D]) #产生一个N*D的数组,元素任意
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0]           

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return result #直接输出抽样  

def LHSample_exp(D,bounds,N):#输出抽样的指数
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''

    result = np.empty([N, D]) #产生一个N*D的数组,元素任意
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0]           

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return np.power(10, result)#输出抽样的指数
###############################################################################
    
def LHSample_n(D,bounds,N):#抽取希尔指数n
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''

    result = np.empty([N, D]) #产生一个N*D的数组,元素任意
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(low=j*d, high=(j+1)*d, size = 1)[0] 

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    #对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错') 
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,(upper_bounds - lower_bounds),out=result),lower_bounds,out=result)
    return np.trunc(result) #执行向下取整操作，元素取整之后的格式依旧不是整数，如果需要保证元素本身就是int数据类型，那么需要添加这条语句：C = B.astype(int)


################################### parameter space################################
#reference:2018-cellsystem-Incoherent Inputs Enhance the Robustness of Biological Oscillators
#para space: kS:0---1    k:10^-1---10^3      kinh:10^-3---10^1      K:10^-3---10^1     n:1---4
#[kS, k1, k2, k3, k4, j1, j2, j3, j4, kinhA, kinhB, n1, n2, n3, n4] =  para

#(0)对 kS 随机赋值(1个参数)   
D_kS = 1        #参数维数
bounds_kS = [[0, 1]]

#(1)对 k1, k2, k3, k4 随机赋值(4个参数)   
D_k = 4        #参数维数
bounds_k = [[-1, 3], [-1, 3], [-1, 3], [-1, 3]]
#para_k = LHSample_exp(D_k, bounds_k, N)

#(2)对 j1, j2, j3, j4, kinhA, kinhB 随机赋值(6个参数) 
D_j = 6   #参数维数
bounds_j = [[-3, 1], [-3, 1], [-3, 1], [-3, 1], [-3, 1], [-3, 1]] #参数范围
#para_j = LHSample_exp(D_j, bounds_j, N)

#(3)对 n1, n2, n3, n4 随机赋值(4个参数) 
D_n = 4   #参数维数
bounds_n = [[1, 5], [1, 5], [1, 5], [1, 5]] #参数范围
#para_n = LHSample_n(D_n, bounds_n, N)
#np.savetxt('J_parameter.csv', para_J, delimiter = ',') #保存参数     
###############################################################################  


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

couple_matrix = [ 1., -1., 1., 0.]   #Motif1-11-1

N = 1000000   #50000#    #取多少组参数
para_kS = LHSample(D_kS, bounds_kS, N)
para_k = LHSample_exp(D_k, bounds_k, N)
para_j = LHSample_exp(D_j, bounds_j, N)
para_n = LHSample_n(D_n, bounds_n, N)
LHS_of_paras = np.hstack((para_kS, para_k, para_j, para_n))

np.savetxt('LHS'+str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3]))+'.csv', LHS_of_paras, delimiter = ',') #保存参数  


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

            
filename = 'Motif' + str(int(couple_matrix[0]))+str(int(couple_matrix[1]))+str(int(couple_matrix[2]))+str(int(couple_matrix[3])) + 'oscill_para.csv'

f=open(filename, 'a')
#rows: response to different parameters(only oscillators are shown)
#columns: from left to right: period, amplitude, FWHM/period, parameters    
    
for j in range(data[0], data[1]): 
    
    if(j%10000==0):
        print([j, N])
    
    t_end = 200
    dt = 0.001
    t = np.arange(0, t_end, dt)
    m = int(0.5*t_end/dt)        #在最后m步通过比较最大值，最小值判断是否震荡
    
    para = LHS_of_paras[j, :]
    
    
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
    
    
    if(np.max(track) <= 1 and np.min(track) >= 0):  #初步解序列有没有越界
        
        NN = track[-m:-1,1]   

        value_oscill = judge_oscill(NN, t_end)
    
        if(type(value_oscill) == tuple):
            #oscill_para.append(all_Para)
            f.write(str(list(value_oscill) + list(para)).replace("[","").replace("]","") + "\n")

f.close() 

MPI.Finalize() 
print(time.time() - t0)   
print('The code have run')
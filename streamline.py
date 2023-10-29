import matplotlib.pyplot as plt
import numpy as np
import math

filename = './v1_v5_H_amplitude_peak_biphase_para.csv'
f1=open(filename,"rb")
case_train=np.loadtxt(f1, delimiter=',', skiprows=0)
f1.close()

case_train = np.array(case_train) 

for j in range(53, 54): 

    para = case_train[j, 3:]
    
    kS = 0.4
    
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
    
    
#    k1 = round(para[1], 1)  #para[1] 
#    k2 = round(para[2], 1)  #para[2]
#    k3 = round(para[3], 1)  #para[3] 
#    k4 = round(para[4], 1)  #para[4]
#    j1 = round(para[5], 1)  #para[5] 
#    j2 = round(para[6], 4)  #para[6] 
#    j3 = round(para[7], 1)  #para[7] 
#    j4 = round(para[8], 1)  #para[8]
#    kinhA = round(para[9], 1)  #para[9] 
#    kinhB = round(para[10], 1)  #para[10] 
    n1 = para[11] 
    n2 = para[12] 
    n3 = para[13] 
    n4 = para[14]


a = np.arange(0, 1.05, 0.1)
b = np.arange(0, 1.05, 0.1)
A, B = np.meshgrid(a, b)


#A = 0.15
#B = 0.1

dA = kS*(1-A)  + k1*A*(1-A)**n1/((1-A)**n1+j1**n1) -k2*B*A**n2/(A**n2+j2**n2) - kinhA*A
dB = k3*A*(1-B)**n3/((1-B)**n3+j3**n3) - kinhB*B
#print(dA, dB)


plt.quiver(A, B, dA, dB,color="black",pivot="tail",units="inches")   #'tail', 'mid', 'middle', 'tip'
plt.scatter(A, B,color="b",s=0.05)
plt.show()




A1 = dA.flatten()
B1 = dB.flatten()
length = np.sqrt(pow(A1, 2) + pow(B1, 2))
angle = []

for k in range(len(A1)):
    if(A1[k]>0):
        angle.append(math.atan(B1[k]/A1[k]))
    else:
        angle.append(math.pi + math.atan(B1[k]/A1[k]))

angle = np.array(angle)
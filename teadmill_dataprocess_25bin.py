# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:17:58 2022

@author: tsiragy
"""

import os
clear = lambda: os.system('cls')  # On Windows System
clear()

import pymatreader

from tkinter import *
from tkinter import filedialog

def get_file_path():
    global file_path
    # Open and return file path
    file_path= filedialog.askopenfilename(title = "Select A File")
    l1 = Label(window, text = "File path: " + file_path).pack()

window = Tk()
# Creating a button to search the file
b = Button(window, text = "Open File", command = get_file_path).pack()
window.mainloop()
print(file_path)

filename = file_path #get the .mat file with the biomech data


window = Tk()
# Creating a button to search the file
b = Button(window, text = "Open File", command = get_file_path).pack()
window.mainloop()
print(file_path)


markers_path = file_path #get the .xlsx file with the rotated marker data

from pymatreader import read_mat
data = read_mat(filename)

COMa = data['data']['COM']['COMa'];
COMp = data['data']['COM']['COMp'];
COMv = data['data']['COM']['COMv'];
TrkX = COMv['torso_X'] 
TrkY = COMv['torso_Y']
TrkZ = COMv['torso_Z']
Trk_AngX = COMv['torso_Ox']
Trk_AngY = COMv['torso_Oy']
Trk_AngZ = COMv['torso_Oz']

keys = ['center_of_mass_X', 'center_of_mass_Y', 'center_of_mass_Z'];

for key in keys:                               
# X is AP, Y is vertical, Z is mediolateral
    COMacc = [COMa.get(key) for key in keys]; #data rows X, Y, Z in order from 0-2
    COMpos = [COMp.get(key) for key in keys]; #data rows X, Y, Z in order from 0-2
    COMvel = [COMv.get(key) for key in keys]; #data rows X, Y, Z in order from 0-2


import numpy as np
import pandas as pd
import scipy
Markers_All = pd.read_excel(markers_path)


Markers_All.columns = list(range(Markers_All.shape[1]))
RLHL_ind = Markers_All.columns[(Markers_All.values=='RLHL').any(0)].tolist()
RLHL_X = Markers_All[RLHL_ind[0]]
RLHL_X = RLHL_X.tolist()
RLHL_Y = Markers_All[RLHL_ind[0]+1]
RLHL_Y = RLHL_Y.tolist()
RLHL_Z = Markers_All[RLHL_ind[0]+2]
RLHL_Z = RLHL_Z.tolist()

RLHL_t = np.transpose(np.array([RLHL_X[4:], RLHL_Y[4:], RLHL_Z[4:]]))

LLHL_ind = Markers_All.columns[(Markers_All.values=='LLHL').any(0)].tolist()
LLHL_X = Markers_All[LLHL_ind[0]]
LLHL_X = LLHL_X.tolist()
LLHL_Y = Markers_All[LLHL_ind[0]+1]
LLHL_Y = LLHL_Y.tolist()
LLHL_Z = Markers_All[LLHL_ind[0]+2]
LLHL_Z = LLHL_Z.tolist()


LLHL_t = np.transpose(np.array([LLHL_X[4:], LLHL_Y[4:], LLHL_Z[4:]]))


RHEE_ind = Markers_All.columns[(Markers_All.values=='RHEE').any(0)].tolist()
RHEE_X = Markers_All[RHEE_ind[0]]
RHEE_X = RHEE_X.tolist()
RHEE_Y = Markers_All[RHEE_ind[0]+1]
RHEE_Y = RHEE_Y.tolist()
RHEE_Z = Markers_All[RHEE_ind[0]+2]
RHEE_Z = RHEE_Z.tolist()

RHEE_t = np.transpose(np.array([RHEE_X[4:], RHEE_Y[4:], RHEE_Z[4:]]))




LHEE_ind = Markers_All.columns[(Markers_All.values=='LHEE').any(0)].tolist()
LHEE_X = Markers_All[LHEE_ind[0]]
LHEE_X = LHEE_X.tolist()
LHEE_Y = Markers_All[LHEE_ind[0]+1]
LHEE_Y = LHEE_Y.tolist()
LHEE_Z = Markers_All[LHEE_ind[0]+2]
LHEE_Z = LHEE_Z.tolist()

LHEE_t = np.transpose(np.array([LHEE_X[4:], LHEE_Y[4:], LHEE_Z[4:]]))




Markers_All.columns = list(range(Markers_All.shape[1]))
RTOE_ind = Markers_All.columns[(Markers_All.values=='RTOE').any(0)].tolist()
RTOE_X = Markers_All[RTOE_ind[0]]
RTOE_X = RTOE_X.tolist()
RTOE_Y = Markers_All[RTOE_ind[0]+1]
RTOE_Y = RTOE_Y.tolist()
RTOE_Z = Markers_All[RTOE_ind[0]+2]
RTOE_Z = RTOE_Z.tolist()

RTOE_t = np.transpose(np.array([RTOE_X[4:], RTOE_Y[4:], RTOE_Z[4:]]))

#RTOE_VT = RTOE_t[:,1]
#RTOE_AP = RTOE_t[:500,0]


#peak_indR = find_peaks(-RTOE_VT, height = -0.15, distance=25)
#TOE_ind = peak_indR[0]
#TOE_indx = TOE_ind[::2]

#plt.plot(-RTOE_VT)
#plt.title('Right Toe Marker Vertical Position')
#plt.plot(TOE_ind, -RTOE_VT[TOE_ind], '^')
#plt.show()


#plt.plot(RTOE_AP)
#plt.title('Right Toe Marker AP Position')
#plt.plot(RHS_indx, RHEE_VT[RHS_indx], '^')
#plt.show()





Last_fr = (len(RLHL_t)-3000)



RLHL = RLHL_t[3000:Last_fr,:]
LLHL = LLHL_t[3000:Last_fr,:]
RHEE = RHEE_t[3000:Last_fr,:]
LHEE = LHEE_t[3000:Last_fr,:]




#test = RHEE[0:1000, :]
#test1 = test[:,0]


#RHEE_diff = np.diff(test1)
#RHEE_diff = np.diff(RHEE[:,1]);
#LHEE_diff = np.diff(LHEE[:,1])
#time = 1/100;
#RHEEv = RHEE_diff/time; #right heel marker velocity
#LHEEv = LHEE_diff/time; #left heel marker velocity

#asign = np.sign(RHEEv)
#signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
#zero_crossings = np.where(np.diff(np.sign(RHEEv)))[0] 
#xx = np.diff(asign)
#RHS_indx = np.where(xx == -2)[0]



#asign_L = np.sign(LHEEv)
#xx_L = np.diff(asign_L)
#LHS_indx = np.where(xx_L == -2)[0] 
 
#RLHL = RLHL.to_numpy(dtype='float', na_value=np.nan)
#LLHL = LLHL.to_numpy(dtype='float', na_value=np.nan)

import matplotlib.pyplot as plt
#%matplotlib qt
from scipy.signal import find_peaks


RHEE_VT = RHEE[:,1] #vertical position of heel marker not velocity
RHEE_vt = (-RHEE_VT)
peak_indR = find_peaks(RHEE_vt, distance=50)
RHS_indx = peak_indR[0]; #Right Heel-strike indices


plt.plot(RHEE_VT)
plt.title('Right Heel Marker Vertical Position')
plt.plot(RHS_indx, RHEE_VT[RHS_indx], '^')
plt.show()



LHEE_VT = LHEE[:,1] #vertical position of heel marker not velocity
LHEE_vt = (-LHEE_VT)
peak_indL = find_peaks(LHEE_vt, distance=50)
LHS_indx = peak_indL[0]; #Right Heel-strike indices


plt.plot(LHEE_VT)
plt.title('Left Heel Marker Vertical Position')
plt.plot(LHS_indx, LHEE_VT[LHS_indx], '^')
plt.show()

time = 1/100
#LHEE_AP = LHEE[:,0]
#LHEE_AP_dif = np.diff(LHEE_AP)
#LHEE_AP_vel = LHEE_AP_dif/time
#test = np.sign(LHEE_AP_vel)
#test1 = np.diff(test)
#searchvalue = 2
#ii = np.where(test1 == searchvalue)[0]






#LHEE_VT = LHEE[:,1] #vertical position of heel marker not velocity
#LHEE_vt = (-LHEE_VT)
#peak_indL = find_peaks(LHEE_vt)
#LHS_indx = peak_indL[0]; #Left Heel-strike indices
#LHS_indx = LHS_indx[LHS_indx>RHS_indx[0]] #keep left heel-strikes that come after the very first right heel-strike



#plt.plot(LHEE_vt)
#plt.title('Left Heel Marker Vertical Position')
#plt.plot(LHS_indx, LHEE_vt[LHS_indx], '^')
#plt.show()


# Find Toe Off Events
#RHEE_VT_Vel = RHEE_VT/time #vertical velocity of heel marker 
#RHEE_vt = (-RHEE_VT)
#Peak_indR_TO = find_peaks(RHEEv, height= 1, distance=100)
#RTO_indx = Peak_indR_TO[0]; #Right Toe-OFF indices
#RTO_indx = RTO_indx[]

#LHEE_VT_Vel = LHEE_VT/time #vertical velocity of the left heel marker
#Peak_indL_TO = find_peaks(LHEEv, height= 1, distance=100)
#LTO_indx = Peak_indL_TO[0];
#LTO_indx = LTO_indx[]



#LHS_indx = LHS_indx[LHS_indx<LTO_indx[-1]]
#LTO_indx = LTO_indx[LTO_indx>LHS_indx[0]]

#RHS_indx = RHS_indx[RHS_indx<RTO_indx[-1]]
#RTO_indx = RTO_indx[RTO_indx>RHS_indx[0]]



#plt.plot(RHEEv)
#plt.title('Right Heel Marker Vertical Velocity')
#plt.plot(RTO_indx, RHEEv[RTO_indx], '^')
#plt.show()

#plt.plot(LHEEv)
#plt.title('Left  Heel Marker Vertical Velocity')
#plt.plot(LTO_indx, LHEEv[LTO_indx], '^')
#plt.show()


#plt.plot(RHEEv)
#plt.title('Right Heel Marker Velocity')
#plt.plot(RHS_indx, RHEEv[RHS_indx], '^')
#plt.show()



#plt.plot(LHEEv)
#plt.title('Left Heel Marker Velocity')
#plt.plot(LHS_indx, LHEEv[LHS_indx], '^')
#plt.show()


#RHv = (-RHEEv)
#peak_ind = find_peaks(RHv, distance=100)
#RHS_indx = peak_ind[0]; #Heel-strike indices


#LHv = (-LHEEv)
#Lpeak_ind = find_peaks(LHv, distance = 100)
#LHS_indx = Lpeak_ind[0];
#LHS_indx = LHS_indx[LHS_indx>RHS_indx[0]] #keep left heel-strikes that come after the very first right heel-strike


#test1 = RHv[peak_indx]
#test2 = [i for i in test1 if i >=0.4]
#test3 = np.nonzero(np.in1d(RHv,test2))[0]

#plt.plot(LHEEv)
#plt.title('Left Heel Marker Velocity')
#plt.plot(LHS_indx, LHEEv[LHS_indx], '^')
#plt.show()




#plt.plot(RHv)
#plt.plot(peak_indx, RHv[peak_indx], '^')
#plt.show()

#plt.plot(RHEEv)
#plt.title('Right Heel Marker Velocity')
#plt.plot(RHS_indx, RHEEv[RHS_indx], '^')
#plt.show()



#time = 1/100   
idx_start = 0
idx_stop = 1

#indx_st = 3
#indx_stp = 4

#frameNumStart = RHS_indx[indx_st]
#frameNumStop = RHS_indx[indx_stp]



COMpos = np.array(COMpos)
COMpos = COMpos.transpose()
COMpos_len = len(COMpos)-3000
COM_pos  = COMpos[3000:COMpos_len,:]


COMvel = np.array(COMvel)
COMvel = COMvel.transpose()
COMvel_len = len(COMvel)-3000
COMvel = COMvel[3000:COMvel_len,:]


TrkX = TrkX[3000:COMvel_len]
TrkY = TrkY[3000:COMvel_len]
TrkZ = TrkZ[3000:COMvel_len]

Trk_AngX = Trk_AngX[3000:COMvel_len]
Trk_AngY = Trk_AngY[3000:COMvel_len]
Trk_Ang =  Trk_AngZ[3000:COMvel_len]

COMacc = np.diff(COMvel, axis=0)/time




#Rtibia_ML_t =  COMv["tibia_r_Y"]
#Rtibia_ML = Rtibia_ML_t[3000:Last_fr]
#Rtibia_ml = (-Rtibia_ML)
#peak_indR = find_peaks(Rtibia_ml, height= 0.21, distance = 10)
#RHS_indx_t = peak_indR[0]; #Right Heel-strike indices
#LHS_indx = LHS_indx[LHS_indx>RHS_indx[0]] #keep left heel-strikes that come after the very first right heel-strike
#RHS_indx = RHS_indx_t[1::2]




#Ltibia_ML_t =  COMv["tibia_l_Y"]
#Ltibia_ML = Ltibia_ML_t[3000:Last_fr]
#Ltibia_ml = (-Ltibia_ML)
##peak_indL = find_peaks(Ltibia_ml, height= 0.21, distance = 10)
#LHS_indx_tt = peak_indL[0]; #Left Heel-strike indices
##LHS_indx_t = LHS_indx_tt[LHS_indx_tt>RHS_indx_t[0]] #keep left heel-strikes that come after the very first right heel-strike
#LHS_indx = LHS_indx_t[1::2]



#plt.plot(Ltibia_ML)
#plt.plot(LHS_indx, Ltibia_ML[LHS_indx], '^')
#plt.show()


#plt.plot(Rtibia_ML)
#plt.plot(RHS_indx_t, Rtibia_ML[RHS_indx_t], '^')
#plt.show()




HS = np.sort(np.concatenate((RHS_indx, LHS_indx), axis=0))
Step_Times = np.diff(HS)
RStep_Times = Step_Times[::2]
RStep_Times = RStep_Times/100



LStep_Times = Step_Times[1::2]
LStep_Times = LStep_Times/100


end = len(RHS_indx);


#plt.plot(COMvel[RHS_indx[3]:RHS_indx[4],0])
#plt.show()


#Trk_Frames = np.arange(len(Trk_AngY))

# Interpolate the data to 101 points for right strides
COMP = pd.DataFrame()
COMV = pd.DataFrame()
COMAcc = pd.DataFrame()
RLHLf = pd.DataFrame()
RHEEf = pd.DataFrame()

for i in range(0, end-1): 
    # Get frames
    frameNumStart = RHS_indx[i]
    frameNumStop = RHS_indx[i+1]
    
    # Get you data
    tmp = scipy.signal.resample(COMpos[frameNumStart:frameNumStop,:], 101)
    tmpV = scipy.signal.resample(COMvel[frameNumStart:frameNumStop,:], 101)
    tmpA = scipy.signal.resample(COMacc[frameNumStart:frameNumStop,:], 101)
    tRLHL = scipy.signal.resample(RLHL[frameNumStart:frameNumStop,:], 101)
    tRHEE = scipy.signal.resample(RHEE[frameNumStart:frameNumStop,:], 101)
    #tmp = scipy.signal.resample(temp, 101)
    COMP = COMP.append(pd.DataFrame([[tmp]]), ignore_index=True)
    COMV = COMV.append(pd.DataFrame([[tmpV]]), ignore_index=True)
    COMAcc = COMAcc.append(pd.DataFrame([[tmpA]]), ignore_index=True)
    RLHLf = RLHLf.append(pd.DataFrame([[tRLHL]]), ignore_index=True)
    RHEEf = RHEEf.append(pd.DataFrame([[tRHEE]]), ignore_index=True)
    #print(i)
   







#Interpolate the data to 101 points for left strides 
end_lft = end = len(LHS_indx);
COMPL = pd.DataFrame()
COMVL = pd.DataFrame()
LLHLf = pd.DataFrame()
LHEEf = pd.DataFrame()


for i in range(0, end_lft-1): 
    # Get frames
    frameNumStart_L = LHS_indx[i]
    frameNumStop_L = LHS_indx[i+1]
    
    # Get you data
    tmp_L = scipy.signal.resample(COMpos[frameNumStart_L:frameNumStop_L,:], 101)
    tmpV_L = scipy.signal.resample(COMvel[frameNumStart_L:frameNumStop_L,:], 101)
    tLLHL = scipy.signal.resample(LLHL[frameNumStart_L:frameNumStop_L,:], 101)
    tLHEE = scipy.signal.resample(LHEE[frameNumStart_L:frameNumStop_L,:], 101)
    #tmp = scipy.signal.resample(temp, 101)
    COMPL = COMPL.append(pd.DataFrame([[tmp_L]]), ignore_index=True) #COM position throughout left strides
    COMVL = COMVL.append(pd.DataFrame([[tmpV_L]]), ignore_index=True) #COM velocity throughout left strides 
    LLHLf = LLHLf.append(pd.DataFrame([[tLLHL]]), ignore_index=True) #LLHL marker throughout left strides
    LHEEf = LHEEf.append(pd.DataFrame([[tLHEE]]), ignore_index=True)






COMAcc = COMAcc.values.tolist()
COMAcc = COMAcc[0:420]
Acc_AP = []
Acc_ML = []
Acc_VT = []
for i in COMAcc:
    Acc_AP.append(i[0][:,0]) #Opensim has X = AP
    Acc_VT.append(i[0][:,1]) #Y = VT
    Acc_ML.append(i[0][:,2]) #Z = ML
    







COMPL = COMPL.values.tolist()
COMP_LHS = []
for i in COMPL:
    COMP_LHS.append(i[0][0,:])
    
COMVL = COMVL.values.tolist()
COMVL_LHS = []
for i in COMVL:
    COMVL_LHS.append(i[0][0,:])


LLHLf = LLHLf.values.tolist()
LLHL_HS = []
for i in LLHLf:
    LLHL_HS.append(i[0][0,:])
    


COMP = COMP.values.tolist()
COMP_HS = []
for i in COMP:
    COMP_HS.append(i[0][0,:]) #COM values at right heel-strike
    
    
COMV = COMV.values.tolist()
COMV_HS = []
for i in COMV:
    COMV_HS.append(i[0][0,:]) #COM velocity values at right heel-strike


RLHLf = RLHLf.values.tolist()
RLHL_HS = []
for i in RLHLf:
    RLHL_HS.append(i[0][0,:])        
        
RHEEf = RHEEf.values.tolist()
RHEE_HS = []
for i in RHEEf:
    RHEE_HS.append(i[0][0,:])      
        
LHEEf = LHEEf.values.tolist()
LHEE_HS = []
for i in LHEEf:
    LHEE_HS.append(i[0][0,:])    


import math
import statistics

def MOS(pCOM, vCOM, BOS):
    pCOM = np.array(pCOM);
    vCOM = np.array(vCOM); 
    BOS = np.array(BOS);
    pCOM_length = len(pCOM);
    vCOM_length = len(vCOM);
    BOS_length = len(BOS);
    if pCOM_length == vCOM_length:
        print('ok')
    else:
        print('Both signals must be equal in length. Please re-check!')
    """Calculate Inverted Pendulum Length
    """
    distances = []
    end1 = len(COMP_HS)
    for i in range(0,end1):
        dist = math.dist(COMP_HS[i], RLHL_HS[i])
        distances.append(dist)

    distances = np.array(distances)
    sqrt_dis = np.sqrt(distances);
    l = statistics.mean(sqrt_dis);
    w = np.sqrt(9.81/l); #calculate the intverted pendulum length 
    
    
    pCOM_AP = pCOM[:, 0];
    pCOM_V = pCOM[:,1];
    pCOM_ML = pCOM[:,2];
    
    
    vCOM_AP = vCOM[:,0];
    vCOM_V = vCOM[:,1]; 
    vCOM_ML = vCOM[:,2];
    
    AP_xCOM = pCOM_AP + (vCOM_AP/w);
    ML_xCOM = pCOM_ML + (vCOM_ML/w);
    
    return AP_xCOM, ML_xCOM 


AP_xCOM_R, ML_xCOM_R = MOS(COMP_HS, COMV_HS, RLHL_HS)
AP_xCOM_L, ML_xCOM_L = MOS(COMP_LHS, COMVL_LHS, LLHL_HS)



RHEE_HS = np.array(RHEE_HS)
RLHL_HS = np.array(RLHL_HS)
LHEE_HS = np.array(LHEE_HS)


MOS_ML_R = RHEE_HS[:,2] - ML_xCOM_R
MOS_AP_R = RHEE_HS[:,0] - AP_xCOM_R

MOS_ML_L = abs(LHEE_HS[:,2] - ML_xCOM_L)
MOS_AP_L = LHEE_HS[:,0] - AP_xCOM_L





#Define and calculate the Harmonic Ratio for all 3 directions (w/ Plots of FFT)
def HR_AP(x, fs):
   x = x #acceleration signal, must be a numpy array 
# x = x.ravel()
   fs = fs;
 #fund_freq = 1 #the fundamental frequency 
   S_length = len(x)
   FFT_aCOG = scipy.fft.fft(x)
   mag_FFT_aCOG = abs(FFT_aCOG)
# avg_FFT = statistics.mean(mag_FFT_aCOG)
   FFT_norm = mag_FFT_aCOG/len(mag_FFT_aCOG)



   FFT_Norm = np.array(FFT_norm)
   P1 = FFT_Norm[0:int(S_length/2)]
   P1 = P1*2;
   lin_space = np.linspace(0, int(S_length/2), num= int(S_length/2), endpoint=True) #create an array that's equal 
   fr = fs * lin_space/int(S_length)

   P1_max = np.max(P1); #find the fundamental frequency
   P1_max_loc = np.where(P1 == np.amax(P1))
   P1_max_loc = P1_max_loc[0]
 #FF_loc = fr[P1_max_loc]

 #nf = (fr[P1_max_loc])/2 #find the nyquist frequency at half the fundamental frequency (P1_max)
 #nf_idx = np.where(fr == nf)
 #nf_idx = nf_idx[0]
   #nf_idx = int(P1_max_loc/2)
   #nf = P1[nf_idx]
   nf_id = np.where((fr>0.5)*(fr<1.5))
   nf_id = np.max(P1[nf_id])
   nf_idx = np.where(P1 == np.amax(nf_id))
   nf_idx = nf_idx[0]
   nf = P1[nf_idx]
   
   


   h3_id = np.where((fr>2.5)*(fr<3.5))
   h3_id = np.max(P1[h3_id])
   h3_idx = np.where(P1 == np.amax(h3_id))
   h3_idx = h3_idx[0]
   h3 = P1[h3_idx]


   h4_id = np.where((fr>3.5)*(fr<4.5))
   h4_id = np.max(P1[h4_id])
   h4_idx = np.where(P1 == np.amax(h4_id))
   h4_idx = h4_idx[0]
   h4 = P1[h4_idx]
 
   h5_id = np.where((fr>4.5)*(fr<5.5))
   h5_id = np.max(P1[h5_id])
   h5_idx = np.where(P1 == np.amax(h5_id))
   h5_idx = h5_idx[0]
   h5 = P1[h5_idx]


   h6_id = np.where((fr>5.5)*(fr<6.5))
   h6_id = np.max(P1[h6_id])
   h6_idx = np.where(P1 == np.amax(h6_id))
   h6_idx = h6_idx[0]
   h6 = P1[h6_idx]


   h7_id = np.where((fr>6.5)*(fr<7.5))
   h7_id = np.max(P1[h7_id])
   h7_idx = np.where(P1 == np.amax(h7_id))
   h7_idx = h7_idx[0]
   h7 = P1[h7_idx]
 
 
   h8_id = np.where((fr>7.5)*(fr<8.5))
   h8_id = np.max(P1[h8_id])
   h8_idx = np.where(P1 == np.amax(h8_id))
   h8_idx = h8_idx[0]
   h8 = P1[h8_idx]


   h9_id = np.where((fr>8.5)*(fr<9.5))
   h9_id = np.max(P1[h9_id])
   h9_idx = np.where(P1 == np.amax(h9_id))
   h9_idx = h9_idx[0]
   h9 = P1[h9_idx]



   h10_id = np.where((fr>9.5)*(fr<10.5))
   h10_id = np.max(P1[h10_id])
   h10_idx = np.where(P1 == np.amax(h10_id))
   h10_idx = h10_idx[0]
   h10 = P1[h10_idx]


   h11_id = np.where((fr>10.5)*(fr<11.5))
   h11_id = np.max(P1[h11_id])
   h11_idx = np.where(P1 == np.amax(h11_id))
   h11_idx = h11_idx[0]
   h11 = P1[h11_idx]
 

   h12_id = np.where((fr>11.5)*(fr<12.5))
   h12_id = np.max(P1[h12_id])
   h12_idx = np.where(P1 == np.amax(h12_id))
   h12_idx = h12_idx[0]
   h12 = P1[h12_idx]
  

   h13_id = np.where((fr>12.5)*(fr<13.5))
   h13_id = np.max(P1[h13_id])
   h13_idx = np.where(P1 == np.amax(h13_id))
   h13_idx = h13_idx[0]
   h13 = P1[h13_idx]

   h14_id = np.where((fr>13.5)*(fr<14.5))
   h14_id = np.max(P1[h14_id])
   h14_idx = np.where(P1 == np.amax(h14_id))
   h14_idx = h14_idx[0]
   h14 = P1[h14_idx]


   h15_id = np.where((fr>14.5)*(fr<15.5))
   h15_id = np.max(P1[h15_id])
   h15_idx = np.where(P1 == np.amax(h15_id))
   h15_idx = h15_idx[0]
   h15 = P1[h15_idx]


   h16_id = np.where((fr>15.5)*(fr<16.5))
   h16_id = np.max(P1[h16_id])
   h16_idx = np.where(P1 == np.amax(h16_id))
   h16_idx = h16_idx[0]
   h16 = P1[h16_idx]



   h17_id = np.where((fr>16.5)*(fr<17.5))
   h17_id = np.max(P1[h17_id])
   h17_idx = np.where(P1 == np.amax(h17_id))
   h17_idx = h17_idx[0]
   h17 = P1[h17_idx]



   h18_id = np.where((fr>17.5)*(fr<18.5))
   h18_id = np.max(P1[h18_id])
   h18_idx = np.where(P1 == np.amax(h18_id))
   h18_idx = h18_idx[0]
   h18 = P1[h18_idx]



   h19_id = np.where((fr>18.5)*(fr<19.5))
   h19_id = np.max(P1[h19_id])
   h19_idx = np.where(P1 == np.amax(h19_id))
   h19_idx = h19_idx[0]
   h19 = P1[h19_idx]



   h20_id = np.where((fr>19.5)*(fr<20.5))
   h20_id = np.max(P1[h20_id])
   h20_idx = np.where(P1 == np.amax(h20_id))
   h20_idx = h20_idx[0]
   h20 = P1[h20_idx]



   plt.plot(fr, P1)
   plt.title('Anteroposterior Harmonics')
   plt.plot(fr[P1_max_loc], P1[P1_max_loc], '^')
   plt.plot(fr[nf_idx],P1[nf_idx], '^')
   plt.plot(fr[h3_idx], P1[h3_idx], '^')
   plt.plot(fr[h4_idx], P1[h4_idx], '^')
   plt.plot(fr[h5_idx], P1[h5_idx], '^')
   plt.plot(fr[h6_idx], P1[h6_idx], '^')
   plt.plot(fr[h7_idx], P1[h7_idx], '^')
   plt.plot(fr[h8_idx], P1[h8_idx], '^')
   plt.plot(fr[h9_idx], P1[h9_idx], '^')
   plt.plot(fr[h10_idx], P1[h10_idx], '^')
   plt.plot(fr[h11_idx], P1[h11_idx], '^')
   plt.plot(fr[h12_idx], P1[h12_idx], '^')
   plt.plot(fr[h13_idx], P1[h13_idx], '^')
   plt.plot(fr[h14_idx], P1[h14_idx], '^')
   plt.plot(fr[h15_idx], P1[h15_idx], '^')
   plt.plot(fr[h16_idx], P1[h16_idx], '^')
   plt.plot(fr[h17_idx], P1[h17_idx], '^')
   plt.plot(fr[h18_idx], P1[h18_idx], '^')
   plt.plot(fr[h19_idx], P1[h19_idx], '^')
   plt.plot(fr[h20_idx], P1[h20_idx], '^')
   plt.show()


   Har_Even = [P1_max, h4, h6, h8, h10, h12, h14, h16, h18, h20]
   Har_Odd = [nf, h3, h5, h7, h9, h11, h13, h15, h17, h19]
   HR = sum(Har_Even)/sum(Har_Odd)
   return Har_Even, Har_Odd, HR


def HR_VT(x, fs):
   x = x #acceleration signal, must be a numpy array 
# x = x.ravel()
   fs = fs;
 #fund_freq = 1 #the fundamental frequency 
   S_length = len(x)
   FFT_aCOG = scipy.fft.fft(x)
   mag_FFT_aCOG = abs(FFT_aCOG)
# avg_FFT = statistics.mean(mag_FFT_aCOG)
   FFT_norm = mag_FFT_aCOG/len(mag_FFT_aCOG)



   FFT_Norm = np.array(FFT_norm)
   P1 = FFT_Norm[0:int(S_length/2)]
   P1 = P1*2;
   lin_space = np.linspace(0, int(S_length/2), num= int(S_length/2), endpoint=True) #create an array that's equal 
   fr = fs * lin_space/int(S_length)

   P1_max = np.max(P1); #find the fundamental frequency
   P1_max_loc = np.where(P1 == np.amax(P1))
   P1_max_loc = P1_max_loc[0]
 #FF_loc = fr[P1_max_loc]

 #nf = (fr[P1_max_loc])/2 #find the nyquist frequency at half the fundamental frequency (P1_max)
 #nf_idx = np.where(fr == nf)
 #nf_idx = nf_idx[0]
   #nf_idx = int(P1_max_loc/2)
   #nf = P1[nf_idx]
   nf_id = np.where((fr>0.5)*(fr<1.5))
   nf_id = np.max(P1[nf_id])
   nf_idx = np.where(P1 == np.amax(nf_id))
   nf_idx = nf_idx[0]
   nf = P1[nf_idx]
   
   


   h3_id = np.where((fr>2.5)*(fr<3.5))
   h3_id = np.max(P1[h3_id])
   h3_idx = np.where(P1 == np.amax(h3_id))
   h3_idx = h3_idx[0]
   h3 = P1[h3_idx]


   h4_id = np.where((fr>3.5)*(fr<4.5))
   h4_id = np.max(P1[h4_id])
   h4_idx = np.where(P1 == np.amax(h4_id))
   h4_idx = h4_idx[0]
   h4 = P1[h4_idx]
 
   h5_id = np.where((fr>4.5)*(fr<5.5))
   h5_id = np.max(P1[h5_id])
   h5_idx = np.where(P1 == np.amax(h5_id))
   h5_idx = h5_idx[0]
   h5 = P1[h5_idx]


   h6_id = np.where((fr>5.5)*(fr<6.5))
   h6_id = np.max(P1[h6_id])
   h6_idx = np.where(P1 == np.amax(h6_id))
   h6_idx = h6_idx[0]
   h6 = P1[h6_idx]


   h7_id = np.where((fr>6.5)*(fr<7.5))
   h7_id = np.max(P1[h7_id])
   h7_idx = np.where(P1 == np.amax(h7_id))
   h7_idx = h7_idx[0]
   h7 = P1[h7_idx]
 
 
   h8_id = np.where((fr>7.5)*(fr<8.5))
   h8_id = np.max(P1[h8_id])
   h8_idx = np.where(P1 == np.amax(h8_id))
   h8_idx = h8_idx[0]
   h8 = P1[h8_idx]


   h9_id = np.where((fr>8.5)*(fr<9.5))
   h9_id = np.max(P1[h9_id])
   h9_idx = np.where(P1 == np.amax(h9_id))
   h9_idx = h9_idx[0]
   h9 = P1[h9_idx]



   h10_id = np.where((fr>9.5)*(fr<10.5))
   h10_id = np.max(P1[h10_id])
   h10_idx = np.where(P1 == np.amax(h10_id))
   h10_idx = h10_idx[0]
   h10 = P1[h10_idx]


   h11_id = np.where((fr>10.5)*(fr<11.5))
   h11_id = np.max(P1[h11_id])
   h11_idx = np.where(P1 == np.amax(h11_id))
   h11_idx = h11_idx[0]
   h11 = P1[h11_idx]
 

   h12_id = np.where((fr>11.5)*(fr<12.5))
   h12_id = np.max(P1[h12_id])
   h12_idx = np.where(P1 == np.amax(h12_id))
   h12_idx = h12_idx[0]
   h12 = P1[h12_idx]
  

   h13_id = np.where((fr>12.5)*(fr<13.5))
   h13_id = np.max(P1[h13_id])
   h13_idx = np.where(P1 == np.amax(h13_id))
   h13_idx = h13_idx[0]
   h13 = P1[h13_idx]

   h14_id = np.where((fr>13.5)*(fr<14.5))
   h14_id = np.max(P1[h14_id])
   h14_idx = np.where(P1 == np.amax(h14_id))
   h14_idx = h14_idx[0]
   h14 = P1[h14_idx]


   h15_id = np.where((fr>14.5)*(fr<15.5))
   h15_id = np.max(P1[h15_id])
   h15_idx = np.where(P1 == np.amax(h15_id))
   h15_idx = h15_idx[0]
   h15 = P1[h15_idx]


   h16_id = np.where((fr>15.5)*(fr<16.5))
   h16_id = np.max(P1[h16_id])
   h16_idx = np.where(P1 == np.amax(h16_id))
   h16_idx = h16_idx[0]
   h16 = P1[h16_idx]



   h17_id = np.where((fr>16.5)*(fr<17.5))
   h17_id = np.max(P1[h17_id])
   h17_idx = np.where(P1 == np.amax(h17_id))
   h17_idx = h17_idx[0]
   h17 = P1[h17_idx]



   h18_id = np.where((fr>17.5)*(fr<18.5))
   h18_id = np.max(P1[h18_id])
   h18_idx = np.where(P1 == np.amax(h18_id))
   h18_idx = h18_idx[0]
   h18 = P1[h18_idx]



   h19_id = np.where((fr>18.5)*(fr<19.5))
   h19_id = np.max(P1[h19_id])
   h19_idx = np.where(P1 == np.amax(h19_id))
   h19_idx = h19_idx[0]
   h19 = P1[h19_idx]



   h20_id = np.where((fr>19.5)*(fr<20.5))
   h20_id = np.max(P1[h20_id])
   h20_idx = np.where(P1 == np.amax(h20_id))
   h20_idx = h20_idx[0]
   h20 = P1[h20_idx]



   plt.plot(fr, P1)
   plt.title('Vertical Harmonics')
   plt.plot(fr[P1_max_loc], P1[P1_max_loc], '^')
   plt.plot(fr[nf_idx],P1[nf_idx], '^')
   plt.plot(fr[h3_idx], P1[h3_idx], '^')
   plt.plot(fr[h4_idx], P1[h4_idx], '^')
   plt.plot(fr[h5_idx], P1[h5_idx], '^')
   plt.plot(fr[h6_idx], P1[h6_idx], '^')
   plt.plot(fr[h7_idx], P1[h7_idx], '^')
   plt.plot(fr[h8_idx], P1[h8_idx], '^')
   plt.plot(fr[h9_idx], P1[h9_idx], '^')
   plt.plot(fr[h10_idx], P1[h10_idx], '^')
   plt.plot(fr[h11_idx], P1[h11_idx], '^')
   plt.plot(fr[h12_idx], P1[h12_idx], '^')
   plt.plot(fr[h13_idx], P1[h13_idx], '^')
   plt.plot(fr[h14_idx], P1[h14_idx], '^')
   plt.plot(fr[h15_idx], P1[h15_idx], '^')
   plt.plot(fr[h16_idx], P1[h16_idx], '^')
   plt.plot(fr[h17_idx], P1[h17_idx], '^')
   plt.plot(fr[h18_idx], P1[h18_idx], '^')
   plt.plot(fr[h19_idx], P1[h19_idx], '^')
   plt.plot(fr[h20_idx], P1[h20_idx], '^')
   plt.show()


   Har_Even = [P1_max, h4, h6, h8, h10, h12, h14, h16, h18, h20]
   Har_Odd = [nf, h3, h5, h7, h9, h11, h13, h15, h17, h19]
   HR = sum(Har_Even)/sum(Har_Odd)
   return Har_Even, Har_Odd, HR


def HR_ML(x, fs):
   x = x #acceleration signal, must be a numpy array 
# x = x.ravel()
   fs = fs;
 #fund_freq = 1 #the fundamental frequency 
   S_length = len(x)
   FFT_aCOG = scipy.fft.fft(x)
   mag_FFT_aCOG = abs(FFT_aCOG)
# avg_FFT = statistics.mean(mag_FFT_aCOG)
   FFT_norm = mag_FFT_aCOG/len(mag_FFT_aCOG)



   FFT_Norm = np.array(FFT_norm)
   P1 = FFT_Norm[0:int(S_length/2)]
   P1 = P1*2;
   lin_space = np.linspace(0, int(S_length/2), num= int(S_length/2), endpoint=True) #create an array that's equal 
   fr = fs * lin_space/int(S_length)

   P1_max = np.max(P1); #find the fundamental frequency
   P1_max_loc = np.where(P1 == np.amax(P1))
   P1_max_loc = P1_max_loc[0]
 #FF_loc = fr[P1_max_loc]

 #nf = (fr[P1_max_loc])/2 #find the nyquist frequency at half the fundamental frequency (P1_max)
 #nf_idx = np.where(fr == nf)
 #nf_idx = nf_idx[0]
   #nf_idx = int(P1_max_loc/2)
   #nf = P1[nf_idx]
   h2_id = np.where((fr>1.5)*(fr<2.5))
   h2_id = np.max(P1[h2_id])
   h2_idx = np.where(P1 == np.amax(h2_id))
   h2_idx = h2_idx[0]
   h2 = P1[h2_idx]
   
   


   h3_id = np.where((fr>2.5)*(fr<3.5))
   h3_id = np.max(P1[h3_id])
   h3_idx = np.where(P1 == np.amax(h3_id))
   h3_idx = h3_idx[0]
   h3 = P1[h3_idx]


   h4_id = np.where((fr>3.5)*(fr<4.5))
   h4_id = np.max(P1[h4_id])
   h4_idx = np.where(P1 == np.amax(h4_id))
   h4_idx = h4_idx[0]
   h4 = P1[h4_idx]
 
   h5_id = np.where((fr>4.5)*(fr<5.5))
   h5_id = np.max(P1[h5_id])
   h5_idx = np.where(P1 == np.amax(h5_id))
   h5_idx = h5_idx[0]
   h5 = P1[h5_idx]


   h6_id = np.where((fr>5.5)*(fr<6.5))
   h6_id = np.max(P1[h6_id])
   h6_idx = np.where(P1 == np.amax(h6_id))
   h6_idx = h6_idx[0]
   h6 = P1[h6_idx]


   h7_id = np.where((fr>6.5)*(fr<7.5))
   h7_id = np.max(P1[h7_id])
   h7_idx = np.where(P1 == np.amax(h7_id))
   h7_idx = h7_idx[0]
   h7 = P1[h7_idx]
 
 
   h8_id = np.where((fr>7.5)*(fr<8.5))
   h8_id = np.max(P1[h8_id])
   h8_idx = np.where(P1 == np.amax(h8_id))
   h8_idx = h8_idx[0]
   h8 = P1[h8_idx]


   h9_id = np.where((fr>8.5)*(fr<9.5))
   h9_id = np.max(P1[h9_id])
   h9_idx = np.where(P1 == np.amax(h9_id))
   h9_idx = h9_idx[0]
   h9 = P1[h9_idx]



   h10_id = np.where((fr>9.5)*(fr<10.5))
   h10_id = np.max(P1[h10_id])
   h10_idx = np.where(P1 == np.amax(h10_id))
   h10_idx = h10_idx[0]
   h10 = P1[h10_idx]


   h11_id = np.where((fr>10.5)*(fr<11.5))
   h11_id = np.max(P1[h11_id])
   h11_idx = np.where(P1 == np.amax(h11_id))
   h11_idx = h11_idx[0]
   h11 = P1[h11_idx]
 

   h12_id = np.where((fr>11.5)*(fr<12.5))
   h12_id = np.max(P1[h12_id])
   h12_idx = np.where(P1 == np.amax(h12_id))
   h12_idx = h12_idx[0]
   h12 = P1[h12_idx]
  

   h13_id = np.where((fr>12.5)*(fr<13.5))
   h13_id = np.max(P1[h13_id])
   h13_idx = np.where(P1 == np.amax(h13_id))
   h13_idx = h13_idx[0]
   h13 = P1[h13_idx]

   h14_id = np.where((fr>13.5)*(fr<14.5))
   h14_id = np.max(P1[h14_id])
   h14_idx = np.where(P1 == np.amax(h14_id))
   h14_idx = h14_idx[0]
   h14 = P1[h14_idx]


   h15_id = np.where((fr>14.5)*(fr<15.5))
   h15_id = np.max(P1[h15_id])
   h15_idx = np.where(P1 == np.amax(h15_id))
   h15_idx = h15_idx[0]
   h15 = P1[h15_idx]


   h16_id = np.where((fr>15.5)*(fr<16.5))
   h16_id = np.max(P1[h16_id])
   h16_idx = np.where(P1 == np.amax(h16_id))
   h16_idx = h16_idx[0]
   h16 = P1[h16_idx]



   h17_id = np.where((fr>16.5)*(fr<17.5))
   h17_id = np.max(P1[h17_id])
   h17_idx = np.where(P1 == np.amax(h17_id))
   h17_idx = h17_idx[0]
   h17 = P1[h17_idx]



   h18_id = np.where((fr>17.5)*(fr<18.5))
   h18_id = np.max(P1[h18_id])
   h18_idx = np.where(P1 == np.amax(h18_id))
   h18_idx = h18_idx[0]
   h18 = P1[h18_idx]



   h19_id = np.where((fr>18.5)*(fr<19.5))
   h19_id = np.max(P1[h19_id])
   h19_idx = np.where(P1 == np.amax(h19_id))
   h19_idx = h19_idx[0]
   h19 = P1[h19_idx]



   h20_id = np.where((fr>19.5)*(fr<20.5))
   h20_id = np.max(P1[h20_id])
   h20_idx = np.where(P1 == np.amax(h20_id))
   h20_idx = h20_idx[0]
   h20 = P1[h20_idx]



   plt.plot(fr, P1)
   plt.title('Mediolateral Harmonics')
   plt.plot(fr[P1_max_loc], P1[P1_max_loc], '^')
   plt.plot(fr[h2_idx],P1[h2_idx], '^')
   plt.plot(fr[h3_idx], P1[h3_idx], '^')
   plt.plot(fr[h4_idx], P1[h4_idx], '^')
   plt.plot(fr[h5_idx], P1[h5_idx], '^')
   plt.plot(fr[h6_idx], P1[h6_idx], '^')
   plt.plot(fr[h7_idx], P1[h7_idx], '^')
   plt.plot(fr[h8_idx], P1[h8_idx], '^')
   plt.plot(fr[h9_idx], P1[h9_idx], '^')
   plt.plot(fr[h10_idx], P1[h10_idx], '^')
   plt.plot(fr[h11_idx], P1[h11_idx], '^')
   plt.plot(fr[h12_idx], P1[h12_idx], '^')
   plt.plot(fr[h13_idx], P1[h13_idx], '^')
   plt.plot(fr[h14_idx], P1[h14_idx], '^')
   plt.plot(fr[h15_idx], P1[h15_idx], '^')
   plt.plot(fr[h16_idx], P1[h16_idx], '^')
   plt.plot(fr[h17_idx], P1[h17_idx], '^')
   plt.plot(fr[h18_idx], P1[h18_idx], '^')
   plt.plot(fr[h19_idx], P1[h19_idx], '^')
   plt.plot(fr[h20_idx], P1[h20_idx], '^')
   plt.show()


   Har_Even = [h2, h4, h6, h8, h10, h12, h14, h16, h18, h20]
   Har_Odd = [P1_max, h3, h5, h7, h9, h11, h13, h15, h17, h19]
   HR = sum(Har_Odd)/sum(Har_Even)
   return Har_Even, Har_Odd, HR




Acc_VT = np.array(Acc_VT)
Acc_VT = Acc_VT.ravel()
#Har_Even_VT, Har_Odd_VT, HR_VT = HR(Acc_VT, 100)

Acc_AP = np.array(Acc_AP)
Acc_AP = Acc_AP.ravel()
#Har_Even_AP, Har_Odd_AP, HR_AP = HR(Acc_AP, 100)
  
Acc_ML = np.array(Acc_ML)
Acc_ML = Acc_ML.ravel()

 #Calculate Lyapunov Exponent 


#from GitPython import clone
#import julia 


from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main
#from julia.api import Main



data_140 = [TrkX[:RHS_indx[140]], TrkY[:RHS_indx[140]], TrkZ[:RHS_indx[140]], Trk_AngX[:RHS_indx[140]], Trk_AngY[:RHS_indx[140]], Trk_AngZ[:RHS_indx[140]]]
TrunkData_140 = np.array(data_140)
TrunkData_140 = np.transpose(TrunkData_140)
Trk_Frames_140 = np.arange(len(TrkY[:RHS_indx[140]]))


data_280 = [TrkX[RHS_indx[140]:RHS_indx[280]], TrkY[RHS_indx[140]:RHS_indx[280]], TrkZ[RHS_indx[140]:RHS_indx[280]], Trk_AngX[RHS_indx[140]:RHS_indx[280]], Trk_AngY[RHS_indx[140]:RHS_indx[280]], Trk_AngZ[RHS_indx[140]:RHS_indx[280]]]
TrunkData_280 = np.array(data_280)
TrunkData_280 = np.transpose(TrunkData_280)
Trk_Frames_280 = np.arange(len(TrkY[RHS_indx[140]:RHS_indx[280]]))


data_420 = [TrkX[RHS_indx[280]:RHS_indx[420]], TrkY[RHS_indx[280]:RHS_indx[420]], TrkZ[RHS_indx[280]:RHS_indx[420]], Trk_AngX[RHS_indx[280]:RHS_indx[420]], Trk_AngY[RHS_indx[280]:RHS_indx[420]], Trk_AngZ[RHS_indx[280]:RHS_indx[420]]]
TrunkData_420 = np.array(data_420)
TrunkData_420 = np.transpose(TrunkData_420)
Trk_Frames_420 = np.arange(len(TrkY[RHS_indx[280]:RHS_indx[420]]))




from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt



Trk_Int_140 = CubicSpline(Trk_Frames_140,TrunkData_140, axis = 0)
Trk_Time_140 = np.linspace(0, len(TrunkData_140), num= 14000)
New_TrkData_140 = Trk_Int_140(Trk_Time_140)
TrkData_Std_140 = np.std(New_TrkData_140, axis = 0)
TrkData_Sum_140 = np.sum(TrkData_Std_140[0:3])
TrkData_Sum2_140 = np.sum(TrkData_Std_140[3:6])

TrkData_Lin_140 = New_TrkData_140[:,0:3]/TrkData_Sum_140
TrkData_Ang_140 = New_TrkData_140[:,3:6]/TrkData_Sum2_140

Trk_Fin_140 = [TrkData_Lin_140, TrkData_Ang_140]
Trk_Fin_140 = np.concatenate((TrkData_Lin_140, TrkData_Ang_140), axis=1)




Trk_Int_280 = CubicSpline(Trk_Frames_280,TrunkData_280, axis = 0)
Trk_Time_280 = np.linspace(0, len(TrunkData_280), num= 14000)
New_TrkData_280 = Trk_Int_280(Trk_Time_280)
TrkData_Std_280 = np.std(New_TrkData_280, axis = 0)
TrkData_Sum_280 = np.sum(TrkData_Std_280[0:3])
TrkData_Sum2_280 = np.sum(TrkData_Std_280[3:6])

TrkData_Lin_280 = New_TrkData_280[:,0:3]/TrkData_Sum_280
TrkData_Ang_280 = New_TrkData_280[:,3:6]/TrkData_Sum2_280

Trk_Fin_280 = [TrkData_Lin_280, TrkData_Ang_280]
Trk_Fin_280 = np.concatenate((TrkData_Lin_280, TrkData_Ang_280), axis=1)





Trk_Int_420 = CubicSpline(Trk_Frames_420,TrunkData_420, axis = 0)
Trk_Time_420 = np.linspace(0, len(TrunkData_420), num= 14000)
New_TrkData_420 = Trk_Int_420(Trk_Time_420)
TrkData_Std_420 = np.std(New_TrkData_420, axis = 0)
TrkData_Sum_420 = np.sum(TrkData_Std_420[0:3])
TrkData_Sum2_420 = np.sum(TrkData_Std_420[3:6])

TrkData_Lin_420 = New_TrkData_420[:,0:3]/TrkData_Sum_420
TrkData_Ang_420 = New_TrkData_420[:,3:6]/TrkData_Sum2_420

Trk_Fin_420 = [TrkData_Lin_420, TrkData_Ang_420]
Trk_Fin_420 = np.concatenate((TrkData_Lin_420, TrkData_Ang_420), axis=1)





from julia import ChaosTools
d_140 = ChaosTools.Dataset(Trk_Fin_140)

r_140 = ChaosTools.genembed(d_140, [0,25])

ks_140 = Main.eval('collect(0:50)')



E_140 = ChaosTools.lyapunov_from_data(r_140, ks_140, w=100, ntype=ChaosTools.NeighborNumber(1), distance= ChaosTools.Euclidean())

X_140 = np.concatenate((np.ones((51,1)), np.arange(51).reshape((51,1))/100), axis=1)

theta_140 = np.linalg.inv(X_140.T.dot(X_140)).dot(X_140.T).dot(E_140)

lya_140 = theta_140[1]





d_280 = ChaosTools.Dataset(Trk_Fin_280)

r_280 = ChaosTools.genembed(d_280, [0,25])

ks_280 = Main.eval('collect(0:50)')



E_280 = ChaosTools.lyapunov_from_data(r_280, ks_280, w=100, ntype=ChaosTools.NeighborNumber(1), distance= ChaosTools.Euclidean())

X_280 = np.concatenate((np.ones((51,1)), np.arange(51).reshape((51,1))/100), axis=1)

theta_280 = np.linalg.inv(X_280.T.dot(X_280)).dot(X_280.T).dot(E_280)

lya_280 = theta_280[1]



d_420 = ChaosTools.Dataset(Trk_Fin_420)

r_420 = ChaosTools.genembed(d_420, [0,25])

ks_420 = Main.eval('collect(0:50)')



E_420 = ChaosTools.lyapunov_from_data(r_420, ks_420, w=100, ntype=ChaosTools.NeighborNumber(1), distance= ChaosTools.Euclidean())

X_420 = np.concatenate((np.ones((51,1)), np.arange(51).reshape((51,1))/100), axis=1)

theta_420 = np.linalg.inv(X_420.T.dot(X_420)).dot(X_420.T).dot(E_420)

lya_420 = theta_420[1]





#Trk_Int = CubicSpline(Trk_Frames,TrunkData, axis = 0)
#Trk_Time = np.linspace(0, len(TrunkData), num= 12500)
#New_TrkData = Trk_Int(Trk_Time)
#TrkData_Std = np.std(New_TrkData, axis = 0)
#TrkData_Sum = np.sum(TrkData_Std[0:3])
#TrkData_Sum2 = np.sum(TrkData_Std[3:6])

#TrkData_Lin = New_TrkData[:,0:3]/TrkData_Sum
#TrkData_Ang = New_TrkData[:,3:6]/TrkData_Sum2

#Trk_Fin = [TrkData_Lin, TrkData_Ang]
#Trk_Fin = np.concatenate((TrkData_Lin, TrkData_Ang), axis=1)


#from julia import ChaosTools

#d = ChaosTools.Dataset(Trk_Fin)

#r = ChaosTools.genembed(d, [0,25])

#ks = Main.eval('collect(0:50)')



#E = ChaosTools.lyapunov_from_data(r, ks, w=100, ntype=ChaosTools.NeighborNumber(1), distance= ChaosTools.Euclidean())

#X = np.concatenate((np.ones((51,1)), np.arange(51).reshape((51,1))/100), axis=1)

#theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(E)

#λ = theta[1]














#Spatiotemporal Parameters 

#STR_tmp = RHS_indx/100 #convert indices back to seconds
#STL_tmp = LHS_indx/100 

#RTO_tmp = RTO_indx/100
#LTO_tmp = LTO_indx/100


#Stp_TR = abs(np.subtract(STR_tmp, RTO_tmp)) #calculate right step times 


#Stp_TL = abs(np.subtract(STL_tmp, LTO_tmp)) #calculate left step times

#STL_tmp = STL_tmp[:len(STR_tmp)]
#ST_R = abs(np.subtract(STL_tmp, STR_tmp))
#ST_L = np.subtract(STR_tmp[1:], STL_tmp[:end-1])




#StepTimes = np.concatenate((STR_tmp, STL_tmp), axis = 0)
#StepTimes = StepTimes.sort(axis=0) #sort step time values by time stamp, odd (and first number) are right steps and even are left steps (beginning with 2nd row)





#Stride_R = np.diff(STR_tmp)
#Stride_L = np.diff(STL_tmp)

#StrLen_R =  abs(np.diff(RHEE_HS[:,1]))
#StrLen_R = StrLen_R*100;

LHEE_RHS = LHEE[RHS_indx[0:425],:]
RHEE_RHS = RHEE[RHS_indx[0:425],:]
StpLen_R = abs(np.subtract(RHEE_RHS[:,0],LHEE_RHS[:,0])) #right step length


RHEE_LHS = RHEE[LHS_indx[0:425],:]
LHEE_LHS = LHEE[LHS_indx[0:425],:]
StpLen_L = abs(np.subtract(LHEE_LHS[:,0],RHEE_LHS[:,0])) #left step length


RLHL_RHS = RLHL[RHS_indx[0:425],:]
LLHL_RHS = LLHL[RHS_indx[0:425],:]
Stp_WidR = abs(np.subtract(RLHL_RHS[:,2],LLHL_RHS[:,2])) #right step widths


RLHL_LHS = RLHL[LHS_indx[0:425],:]
LLHL_LHS = LLHL[LHS_indx[0:425],:]
Stp_WidL = abs(np.subtract(RLHL_LHS[:,2],LLHL_LHS[:,2])) #left step widths




Bin_Num = int(len(RHS_indx)/25) #number of bins based on 25 strides each
RHS_len = len(RHS_indx)

test_indx = np.zeros((25, Bin_Num))
for i in range(Bin_Num):
    test_indx[:, i] = RHS_indx[i*25:(i+1)*25]



MOS_bin = int(len(MOS_ML_R[:425])/25)
MOS_ML_Rgt = np.zeros((25, MOS_bin))
MOS_ML_Lft = np.zeros((25, MOS_bin))
MOS_AP_Rgt = np.zeros((25, MOS_bin))
MOS_AP_Lft = np.zeros((25, MOS_bin))


for i in range(MOS_bin):
    MOS_ML_Rgt[:, i] = MOS_ML_R[i*25:(i+1)*25]
    MOS_ML_Lft[:, i] = MOS_ML_L[i*25:(i+1)*25]
    MOS_AP_Rgt[:, i] = MOS_AP_R[i*25:(i+1)*25]
    MOS_AP_Lft[:, i] = MOS_AP_L[i*25:(i+1)*25]


RHS_bin = int(len(RHS_indx[:425])/25)
RHS_ind = np.zeros((25, RHS_bin))
#RTO_ind = np.zeros((25, RHS_bin))


LHS_bin = int(len(LHS_indx[:425])/25)
LHS_ind = np.zeros((25, LHS_bin))
#LTO_ind = np.zeros((25, LHS_bin))

Stp_WidR = Stp_WidR[0:425]
Stp_WidL = Stp_WidL[0:425]
StpLen_R = StpLen_R[0:425]
StpLen_L = StpLen_L[0:425]
Step_W_R = np.zeros((25, LHS_bin))
Step_W_L = np.zeros((25, LHS_bin))
Step_LN_R = np.zeros((25, LHS_bin))
Step_LN_L = np.zeros((25, LHS_bin))



StpT_R = np.zeros((25, LHS_bin))
StpT_L =  np.zeros((25, LHS_bin))

RStep_Times = RStep_Times[:425]
LStep_Times = LStep_Times[:425]

for i in range(RHS_bin):
    StpT_R[:,i] = RStep_Times[i*25:(i+1)*25]
    RHS_ind[:, i] = RHS_indx[i*25:(i+1)*25]
   # RTO_ind[:, i] = RTO_indx[i*25:(i+1)*25]
    StpT_L[:,i] = LStep_Times[i*25:(i+1)*25]
    LHS_ind[:, i] = LHS_indx[i*25:(i+1)*25]
    #LTO_ind[:, i] = LTO_indx[i*25:(i+1)*25]
    Step_W_R[:,i] = Stp_WidR[i*25:(i+1)*25]
    Step_W_L[:,i] = Stp_WidL[i*25:(i+1)*25]
    Step_LN_R[:,i] = StpLen_R[i*25:(i+1)*25]
    Step_LN_L[:,i] = StpLen_L[i*25:(i+1)*25]

#StpT_R = np.subtract(RTO_ind, RHS_ind)
#StpT_R = StpT_R/100

#StpT_L = np.subtract(LTO_ind, LHS_ind)
#StpT_L = StpT_L/100








avg_StpT_R = np.zeros((1, LHS_bin))
avg_StpT_L = np.zeros((1, LHS_bin))
avg_StpL_R = np.zeros((1, LHS_bin))
avg_StpL_L = np.zeros((1, LHS_bin))
avg_StpW_R = np.zeros((1, LHS_bin))
avg_StpW_L = np.zeros((1, LHS_bin))
avg_MOS_MLR = np.zeros((1, LHS_bin))
avg_MOS_MLL = np.zeros((1, LHS_bin))
avg_MOS_APR = np.zeros((1, LHS_bin))
avg_MOS_APL = np.zeros((1, LHS_bin))


std_StpT_R = np.zeros((1, LHS_bin))
std_StpT_L = np.zeros((1, LHS_bin))
std_StpL_R = np.zeros((1, LHS_bin))
std_StpL_L = np.zeros((1, LHS_bin))
std_StpW_R = np.zeros((1, LHS_bin))
std_StpW_L = np.zeros((1, LHS_bin))
std_MOS_MLR = np.zeros((1, LHS_bin))
std_MOS_MLL = np.zeros((1, LHS_bin))
std_MOS_APR = np.zeros((1, LHS_bin))
std_MOS_APL = np.zeros((1, LHS_bin))

for i in range(RHS_bin):
    avg_StpT_R[:,i] = np.average(StpT_R[:,i])
    avg_StpT_L[:,i] = np.average(StpT_L[:,i])
    avg_StpL_R[:,i] = np.average(Step_LN_R[:,i])
    avg_StpL_L[:,i] = np.average(Step_LN_L[:,i])
    avg_StpW_R[:,i] = np.average(Step_W_R[:,i])
    avg_StpW_L[:,i] = np.average(Step_W_L[:,i])
    avg_MOS_MLR[:,i] = np.average(MOS_ML_Rgt[:,i])
    avg_MOS_MLL[:,i] = np.average(MOS_ML_Lft[:,i])
    avg_MOS_APR[:,i] = np.average(MOS_AP_Rgt[:,i])
    avg_MOS_APL[:,i] = np.average(MOS_AP_Lft[:,i])  
    
    
    std_StpT_R[:,i] = np.std(StpT_R[:,i])
    std_StpT_L[:,i] = np.std(StpT_L[:,i])
    std_StpL_R[:,i] = np.std(Step_LN_R[:,i])
    std_StpL_L[:,i] = np.std(Step_LN_L[:,i])
    std_StpW_R[:,i] = np.std(Step_W_R[:,i])
    std_StpW_L[:,i] = np.std(Step_W_L[:,i])
    std_MOS_MLR[:,i] = np.std(MOS_ML_Rgt[:,i])
    std_MOS_MLL[:,i] = np.std(MOS_ML_Lft[:,i])
    std_MOS_APR[:,i] = np.std(MOS_AP_Rgt[:,i])
    std_MOS_APL[:,i] = np.std(MOS_AP_Lft[:,i]) 

#Calculate Coefficient of Variation
COV_StpT_R = (std_StpT_R/avg_StpT_R)*100
COV_StpT_L = (std_StpT_L/avg_StpT_L)*100
COV_StpL_R = (std_StpL_R/avg_StpL_R)*100
COV_StpL_L = (std_StpL_L/avg_StpL_L)*100
COV_SW_R = (std_StpW_R/avg_StpW_R)*100
COV_SW_L = (std_StpW_L/avg_StpW_L)*100




Acc_Bin= int(len(Acc_AP)/14140) #number of bins based on 25 strides each per 100 points per stride
#RHS_len = len(RHS_indx)

Acc_APind = np.zeros((14140, Acc_Bin))
Acc_VTind = np.zeros((14140, Acc_Bin))
Acc_MLind = np.zeros((14140, Acc_Bin))


for i in range(0,Acc_Bin):
    Acc_APind[:, i] = Acc_AP[i*14140:(i+1)*14140]
    Acc_VTind[:, i] = Acc_VT[i*14140:(i+1)*14140]
    Acc_MLind[:, i] = Acc_ML[i*14140:(i+1)*14140]
    
    
AccAP_indxR, AccAP_indxC = Acc_APind.shape

HR_AP_EV = np.zeros((10, AccAP_indxC))
HR_AP_Odd = np.zeros((10, AccAP_indxC))
HR_AP_Num = np.zeros((1, AccAP_indxC))



HR_VT_EV = np.zeros((10, AccAP_indxC))
HR_VT_Odd = np.zeros((10, AccAP_indxC))
HR_VT_Num = np.zeros((1, AccAP_indxC))


HR_ML_EV = np.zeros((10, AccAP_indxC))
HR_ML_Odd = np.zeros((10, AccAP_indxC))
HR_ML_Num = np.zeros((1, AccAP_indxC))




for i in range(0,AccAP_indxC):
    HR_AP_EV[:,i], HR_AP_Odd[:,i], HR_AP_Num[:,i] = HR_AP(Acc_APind[:,i], 100)
    HR_VT_EV[:,i], HR_VT_Odd[:,i], HR_VT_Num[:,i] = HR_VT(Acc_VTind[:,i], 100)
    HR_ML_EV[:,i], HR_ML_Odd[:,i], HR_ML_Num[:,i] = HR_ML(Acc_MLind[:,i], 100)

#import pickle

#def save(filename, *args):
    # Get global dictionary
 #   glob = globals()
   # d = {}
  #  for v in args:
        # Copy over desired values
   #     d[v] = glob[v]
    #with open(filename, 'wb') as f:
        # Put them in the file 
     #   pickle.dump(d, f)



#size = len(filename)
filenames = filename[-11:] 
#filenames = filename[:size - 4]
#filenames = markers_path[size-15:size - 8]
#filenames = "S003_S1.mat"

#filenames = markers_path[-12:]
mydict = {'avg_MOS_APL': avg_MOS_APL, 'avg_MOS_APR': avg_MOS_APR,'avg_MOS_MLL':avg_MOS_MLL, 'avg_MOS_MLR': avg_MOS_MLR, 'avg_StpL_L': avg_StpL_L,'avg_StpL_R': avg_StpL_R, 'avg_StpT_L': avg_StpT_L,'avg_StpT_R': avg_StpT_R, 'avg_StpW_L': avg_StpW_L, 'avg_StpW_R': avg_StpW_R,'std_MOS_APL': std_MOS_APL, 'std_MOS_APR': std_MOS_APR,'std_MOS_MLL': std_MOS_MLL, 'std_MOS_MLR': std_MOS_MLR, 'std_StpL_L': std_StpL_L,'std_StpL_R': std_StpL_R, 'std_StpT_L': std_StpT_L,'std_StpT_R': std_StpT_R, 'std_StpW_L': std_StpW_L, 'std_StpW_R': std_StpW_R, 'HR_AP_Num': HR_AP_Num, 'HR_ML_Num': HR_ML_Num, 'HR_VT_Num':HR_VT_Num, 'lya_140': lya_140, 'lya_280': lya_280, 'lya_420': lya_420, 'COV_StpT_R': COV_StpT_R, 'COV_StpT_L':COV_StpT_L, 'COV_StpL_R': COV_StpL_R, 'COV_StpL_L': COV_StpL_L, 'COV_SW_R': COV_SW_R, 'COV_SW_L': COV_SW_L}

from scipy.io import savemat
savemat(filenames, mydict)



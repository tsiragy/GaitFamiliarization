# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:17:48 2022

@author: tsiragy
"""



import os
from os import chdir, getcwd
path=getcwd()
#chdir(wd)
 
# Get the list of all files and directories
#pth = "C:\\Users\\tsiragy\\Documents\\Treadmill Familiarization\\Treadmill Dataset\\Subject_02\\Session_01"
#path ="C:\\Users\\tsiragy\\Documents\\Treadmill Familiarization\\Treadmill Dataset\\Subject_02\\Session_02"
#path = wd;

dir_list = os.listdir(path)
 
print("Files and directories in '", path, "' :")
 
# prints all files
print(dir_list)




markers = "familiarize.xlsx";
Body_acc = "subject-2_BodyKinematics_acc_global.sto";
Body_pos = "subject-2_BodyKinematics_pos_global.sto";
Body_vel = "subject-2_BodyKinematics_vel_global.sto";
Kin_dud = "subject-2_Kinematics_dudt.sto";
Kin_q = "subject-2_Kinematics_q.sto";
Kin_u = "subject-2_Kinematics_u.sto";
Pkin_acc = "subject-2_PointKinematics_NONAME_acc.sto"; 
Pkin_pos = "subject-2_PointKinematics_NONAME_pos.sto";
Pkin_vel = "subject-2_PointKinematics_NONAME_vel.sto";


#from ezc3d import c3d

#c = c3d(c3d_data);
#point_data = c['data']['points']
#analog_data = c['data']['analogs']

#import numpy as np
#from pyomeca import Markers

#markers = Markers.from_c3d(c3d_data, prefix_delimiter=":")
#markers.isel(channel=9, time=0)

#data_path = "C:\\Users\\tsiragy\\Documents\\Treadmill Familiarization\\Treadmill Dataset\\Subject_02\\Session_01\\familiarize.c3d"

#channels = ["RFHD"]
#channels2 = ["LFHD", "RBHD"]
#"RBHD", "RSHO", "LSHO", "C7", "STRN", "XYPH", "RUA1", "RUA2", "RUA3", "RFAM", "RFAL", "RWRU", "RWRR", "RFIN", "LUA1", "LUA2", "LUA3", "LFAL", "LFAM", "LWRR", "LWRU", "LFIN", "RASI", "LASI", "RASI_2", "LASI_2", "RPSI", "LPSI", "RTH1", "RTH2", "RTH3", "RTH4", "RSK1", "RSK2", "RSK3", "RSK4", "RTOE", "R5MT", "RLHL", "RHEE", "LTH1", "LTH2", "LTH3", "LTH4", "LSK1", "LSK2", "LSK3", "LSK4", "LTOE", "L5MT", "LLHL", "LHEE", "T8", "RBAC"]
#Markers = Markers.from_c3d(c3d_data, usecols=channels)

#RFHD = markers.data[0:3,0,:]




import pyomeca
from pyomeca import Markers





markers_path = path + "\\" + markers;
COM_acc = path + "\\" + Body_acc;
COM_pos = path + "\\" + Body_pos; 
COM_vel = path + "\\" + Body_vel; 
Kin_D = path + "\\" + Kin_dud; 
Kin_Q = path + "\\" + Kin_q;
Kin_U = path + "\\" + Kin_u;
PKin_Acc = path + "\\" + Pkin_acc;
PKin_Pos = path + "\\" + Pkin_pos;
PKin_Vel = path + "\\" + Pkin_vel; 




def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data


#import xml.etree.ElementTree as ET 
#tree = ET.parse(Markers);
#root = tree.getroot();





COM_Acc_header, COM_Acc_labels, COM_Acc_Dt = readMotionFile(COM_acc);
COM_Pos_header, COM_Pos_labels, COM_Pos_Dt = readMotionFile(COM_pos);
COM_Vel_header, COM_Vel_labels, COM_Vel_Dt = readMotionFile(COM_vel);

Kind_header, Kind_labels, Kind_dt = readMotionFile(Kin_D);
Kinq_header, Kinq_labels, Kinq_dt = readMotionFile(Kin_Q);
Kinu_header, Kinu_labels, Kinu_dt = readMotionFile(Kin_U);

PkinA_header, PkinA_labels, PkinA_dt = readMotionFile(PKin_Acc);
PKinP_header, PKinP_labels, PkinP_dt = readMotionFile(PKin_Pos);
PKinV_header, PKinV_labels, PKinV_dt = readMotionFile(PKin_Vel);




import pandas as pd
Markers_All = pd.read_excel(markers_path)
#print(Markers_All)


                               


del Body_vel
del Body_acc
del Body_pos
del COM_acc
del COM_pos
del COM_vel
del Kin_D
del Kin_dud
del Kin_q
del Kin_Q
del Kin_u
del Kin_U
del path
del Pkin_acc
del PKin_Acc
del Pkin_pos
del PKin_Pos
del Pkin_vel
del PKin_Vel
del dir_list
del markers 
del markers_path
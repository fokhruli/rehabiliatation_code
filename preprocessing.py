#exercise = m08
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.decomposition import PCA
import scipy.stats as stats


#file = open('./Segmented_Movements/Kinect/Positions/m08_s01_e01_positions.txt')

#lines = file.readlines()


#path = ["./Segmented_Movements/Kinect/Positions/","./Incorrect_Segmented_Movements/Kinect/Positions/"]

## read all file for m08
def get_path(k):
    correct_movement_location = []
    #label1 = [1]*100
    for i in range(10):
        for j in range(10):
            if i>=9 and j<9:
                correct_movement_location.append("./Segmented_Movements/Kinect/Positions/"+"m08_s{}_e0{}_positions.txt".format(i+1, j+1))
            elif i<9 and j>=9:
                correct_movement_location.append("./Segmented_Movements/Kinect/Positions/"+"m08_s0{}_e{}_positions.txt".format(i+1, j+1))
            elif i>=9 and j>=9:
                correct_movement_location.append("./Segmented_Movements/Kinect/Positions/"+"m08_s{}_e{}_positions.txt".format(i+1, j+1))
            else:
                correct_movement_location.append("./Segmented_Movements/Kinect/Positions/"+"m08_s0{}_e0{}_positions.txt".format(i+1, j+1))
        
    
    incorrect_movement_location = []
    #label2 = [0]*100
    for i in range(10):
        for j in range(10):
            if i>=9 and j<9:
                incorrect_movement_location.append("./Incorrect_Segmented_Movements/Kinect/Positions/"+"m08_s{}_e0{}_positions_inc.txt".format(i+1, j+1))
            elif i<9 and j>=9:
                incorrect_movement_location.append("./Incorrect_Segmented_Movements/Kinect/Positions/"+"m08_s0{}_e{}_positions_inc.txt".format(i+1, j+1))
            elif i>=9 and j>=9:
                incorrect_movement_location.append("./Incorrect_Segmented_Movements/Kinect/Positions/"+"m08_s{}_e{}_positions_inc.txt".format(i+1, j+1))
            else:
                incorrect_movement_location.append("./Incorrect_Segmented_Movements/Kinect/Positions/"+"m08_s0{}_e0{}_positions_inc.txt".format(i+1, j+1))
                
    correct_movement_location.extend(incorrect_movement_location)
    files = open(correct_movement_location[i])
    lines = files.readlines()
    if k<=100:
        label = 1
    else:
        label = 0
    return lines, label

#df, label = get_path(i)

def preprocess(i):
    lines, label = get_path(i)
    for i, line in enumerate(lines):
        lines[i] = line.split(',')
    for i in range(len(lines)):
        lines[i] = [ float(x) for x in  lines[i]]
        
    #lines = np.array(lines)
    columns = []
    
    for i in range(22):
        columns.append("x{}".format(i+1))
        columns.append("y{}".format(i+1))
        columns.append("z{}".format(i+1))
    df1 = pd.DataFrame(data = lines, columns = columns)
    rom = [6, 10, 14, 9, 13, 12, 8, 19, 15]
    #rom_full = ['hd', 'wr_l', 'wr_r', 'eb_l', 'eb_r', 'sh_r', 'sh_l', 'hp_r', 'hp_l']
    

    '''
    data = []
    data = np.array(data)
    for i, r in enumerate(rom):
        
        data.append(lines[:,3*r-3])
        data.append(lines[:,3*r-2])
        data.append(lines[:,3*r-1])
        '''
    col = []
    #col_full = defaultdict()
    for i,r in enumerate(rom):
        col.append("x{}".format(r))
        col.append("y{}".format(r))
        col.append("z{}".format(r))
        
        
        #col_full[rf+"_x"] = "x{}".format(r)
        #col_full[rf+"_y"] = "y{}".format(r)
        #col_full[rf+"_z"] = "z{}".format(r)
        
    
    df = df1[col]
    # make a condition 
    #df['label'] = pd.DataFrame(np.zeros((len(lines),1)))
    return df, label

#df, label = preprocess(i)

def feature_extruction(i):

    df, label = preprocess(i)
    ###################### make ja_t(hp_r, sh_r, eb_r) ##########################
    ## 2*1 col features
    pt_hp_r_sh_r = df[['x19', 'y19', 'z19']].to_numpy() - df[['x12', 'y12', 'z12']].to_numpy()
    pt_hp_l_sh_l = df[['x15', 'y15', 'z15']].to_numpy() - df[['x8', 'y8', 'z8']].to_numpy()
    pt_sh_r_eb_r = df[['x12', 'y12', 'z12']].to_numpy() - df[['x13', 'y13', 'z13']].to_numpy()
    pt_sh_l_eb_l = df[['x8', 'y8', 'z8']].to_numpy() - df[['x9', 'y9', 'z9']].to_numpy()
    
    
    ja_t_hp_r_sh_r_eb_r = []
    ja_t_hp_l_sh_l_eb_l = []
    
    
    for i in range(len(pt_hp_r_sh_r)):
        ja_t_hp_r_sh_r_eb_r.append(np.arccos(np.dot(pt_hp_r_sh_r[i], pt_sh_r_eb_r[i])/(sum(abs(pt_hp_r_sh_r[i]))*sum(abs(pt_sh_r_eb_r[i])))))
        ja_t_hp_l_sh_l_eb_l.append(np.arccos(np.dot(pt_hp_l_sh_l[i], pt_sh_l_eb_l[i])/(sum(abs(pt_hp_l_sh_l[i]))*sum(abs(pt_sh_l_eb_l[i])))))
    
    ja_t_hp_r_sh_r_eb_r = np.array(ja_t_hp_r_sh_r_eb_r).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    ja_t_hp_l_sh_l_eb_l = np.array(ja_t_hp_l_sh_l_eb_l).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    
    features = None
    features = np.append(ja_t_hp_r_sh_r_eb_r,ja_t_hp_l_sh_l_eb_l, axis=1)
    
    ########################## make nrt_t_b_s #######################
    ## 4*1 col features
    
    
    
    rt_t_hd_eb_r = np.sqrt(np.sum(np.square(df[['x6', 'y6', 'z6']].to_numpy() - df[['x13', 'y13', 'z13']].to_numpy()), axis=1))
    rt_t_hd_eb_l = np.sqrt(np.sum(np.square(df[['x6', 'y6', 'z6']].to_numpy() - df[['x9', 'y9', 'z9']].to_numpy()), axis=1))
    rt_t_hd_wr_r = np.sqrt(np.sum(np.square(df[['x6', 'y6', 'z6']].to_numpy() - df[['x14', 'y14', 'z14']].to_numpy()), axis=1))
    rt_t_hd_wr_l = np.sqrt(np.sum(np.square(df[['x6', 'y6', 'z6']].to_numpy() - df[['x10', 'y10', 'z10']].to_numpy()), axis=1))
    
    nrt_t_hd_eb_r = (abs(rt_t_hd_eb_r - rt_t_hd_eb_r[0])/rt_t_hd_eb_r[0]).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    nrt_t_hd_eb_l = (abs(rt_t_hd_eb_l - rt_t_hd_eb_l[0])/rt_t_hd_eb_l[0]).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    nrt_t_hd_wr_r = (abs(rt_t_hd_wr_r - rt_t_hd_wr_r[0])/rt_t_hd_wr_r[0]).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    nrt_t_hd_wr_l = (abs(rt_t_hd_wr_l - rt_t_hd_wr_l[0])/rt_t_hd_wr_l[0]).reshape((len(ja_t_hp_r_sh_r_eb_r),1))
    
    features = np.append(features,nrt_t_hd_eb_r, axis=1)
    features = np.append(features,nrt_t_hd_eb_l, axis=1)
    features = np.append(features,nrt_t_hd_wr_r, axis=1)
    features = np.append(features,nrt_t_hd_wr_l, axis=1)
    
    
    ###########################     #################################
    ## total 4*3 col features
    
    pt_t_hd_wr_r = abs(df[['x6', 'y6', 'z6']].to_numpy() - df[['x14', 'y14', 'z14']].to_numpy())
    pt_t_hd_wr_l = abs(df[['x6', 'y6', 'z6']].to_numpy() - df[['x10', 'y10', 'z10']].to_numpy())
    pt_t_sh_r_wr_r = abs(df[['x12', 'y12', 'z12']].to_numpy() - df[['x14', 'y14', 'z14']].to_numpy())
    pt_t_sh_l_wr_l = abs(df[['x8', 'y8', 'z8']].to_numpy() - df[['x10', 'y10', 'z10']].to_numpy())
    
    
    # devided by zero may occour because there have chance any sensor data is getting zero
    # so we take only differences not doing normalize
    
    #npt_t_hd_wr_r = abs(pt_t_hd_wr_r - pt_t_hd_wr_r[0])/pt_t_hd_wr_r[0]
    #npt_t_hd_wr_l = abs(pt_t_hd_wr_l - pt_t_hd_wr_l[0])/pt_t_hd_wr_l[0]
    #npt_t_sh_r_wr_r = abs(pt_t_sh_r_wr_r - pt_t_sh_r_wr_r[0])/pt_t_sh_r_wr_r[0]
    #npt_t_sh_l_wr_l = abs(pt_t_sh_l_wr_l - pt_t_sh_l_wr_l[0])/pt_t_sh_l_wr_l[0]
    
    npt_t_hd_wr_r = abs(pt_t_hd_wr_r - pt_t_hd_wr_r[0])
    npt_t_hd_wr_l = abs(pt_t_hd_wr_l - pt_t_hd_wr_l[0])
    npt_t_sh_r_wr_r = abs(pt_t_sh_r_wr_r - pt_t_sh_r_wr_r[0])
    npt_t_sh_l_wr_l = abs(pt_t_sh_l_wr_l - pt_t_sh_l_wr_l[0]) 
    
    features = np.append(features,npt_t_hd_wr_r, axis=1)
    features = np.append(features,npt_t_hd_wr_l, axis=1)
    features = np.append(features,npt_t_sh_r_wr_r, axis=1)
    features = np.append(features,npt_t_sh_l_wr_l, axis=1)
    #features = np.append(npt_t_hd_wr_r,npt_t_hd_wr_l, axis=1)
    if label==1:
        label = np.ones((len(pt_t_hd_wr_r), 1))
    elif label==0:
        label = np.zeros((len(pt_t_hd_wr_r), 1))
    data = np.append(features, label, axis=1)
    return data

def load_data():
    for i in range(200):
        temp = feature_extruction(i)
        if i == 0:
            dataframe = temp
        else:
            dataframe = np.append(dataframe,temp, axis=0)
    col = []
#col_full = defaultdict()
    for i in range(19):
        col.append("x{}".format(i))
    dataframe = pd.DataFrame(data = dataframe, columns = col)
    return dataframe

d = load_data()

Fs = 10
frame_size = Fs*2 # 20
hop_size = Fs*1 # 10



def get_frames(frame_size, hop_size):
    df = load_data()
    N_FEATURES = 18

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x0 = df['x0'].values[i: i + frame_size]
        x1 = df['x1'].values[i: i + frame_size]
        x2 = df['x2'].values[i: i + frame_size]
        x3 = df['x3'].values[i: i + frame_size]
        x4 = df['x4'].values[i: i + frame_size]
        x5 = df['x5'].values[i: i + frame_size]
        x6 = df['x6'].values[i: i + frame_size]
        x7 = df['x7'].values[i: i + frame_size]
        x8 = df['x8'].values[i: i + frame_size]
        x9 = df['x9'].values[i: i + frame_size]
        x10 = df['x10'].values[i: i + frame_size]
        x11 = df['x11'].values[i: i + frame_size]
        x12 = df['x12'].values[i: i + frame_size]
        x13 = df['x13'].values[i: i + frame_size]
        x14 = df['x14'].values[i: i + frame_size]
        x15 = df['x15'].values[i: i + frame_size]
        x16 = df['x16'].values[i: i + frame_size]
        x17 = df['x17'].values[i: i + frame_size]
        #y = df['y'].values[i: i + frame_size]
        #z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['x18'][i: i + frame_size])[0][0]
        frames.append([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16,x17])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels
def get_data():
    X,y =get_frames(frame_size, hop_size)
    X_ = [0]*len(X)
    for i in range(len(X)):
        X_[i] = X[i].flatten()
    return np.array(X_), y
    
#X_iii, y = get_data()

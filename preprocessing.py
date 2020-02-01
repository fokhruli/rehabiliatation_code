#exercise = m08
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.decomposition import PCA


#file = open('./Segmented_Movements/Kinect/Positions/m08_s01_e01_positions.txt')

#lines = file.readlines()


#path = ["./Segmented_Movements/Kinect/Positions/","./Incorrect_Segmented_Movements/Kinect/Positions/"]

## read all file for m08
def get_path():
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
    p = random.randint(0,200)
    files = open(correct_movement_location[p])
    lines = files.readlines()
    if p<=100:
        label = 0
    else:
        label = 1
    if len(lines)>=50:
        return lines, label
    else:
        preprocess()

def preprocess():
    lines, label = get_path()
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


def feature_extruction():

    df, label = preprocess()
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
    return features.flatten(), label


#feature_extruction()

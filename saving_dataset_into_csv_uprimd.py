#exercise = m08
import pandas as pd
import numpy as np
from collections import defaultdict
import random
from sklearn.decomposition import PCA
import scipy.stats as stats
from numpy import savetxt


#read all file for correct movement
dataset_correct = pd.read_csv("Segmented Movements/Kinect/Positions/m08_s01_e01_positions.txt").to_numpy()
for i in range(0,10):
    for j in range(1,10):
        if i>=9 and j<9:
            dataset_correct_ = pd.read_csv("Segmented Movements/Kinect/Positions/"+"m08_s{}_e0{}_positions.txt".format(i+1, j+1)).to_numpy()
            dataset_correct = np.vstack((dataset_correct,dataset_correct_))
        elif i<9 and j>=9:
            dataset_correct_ = pd.read_csv("Segmented Movements/Kinect/Positions/"+"m08_s0{}_e{}_positions.txt".format(i+1, j+1)).to_numpy()
            dataset_correct = np.vstack((dataset_correct,dataset_correct_))
        elif i>=9 and j>=9:
            dataset_correct_ = pd.read_csv("Segmented Movements/Kinect/Positions/"+"m08_s{}_e{}_positions.txt".format(i+1, j+1)).to_numpy()
            dataset_correct = np.vstack((dataset_correct,dataset_correct_))
        else:
            dataset_correct_ = pd.read_csv("Segmented Movements/Kinect/Positions/"+"m08_s0{}_e0{}_positions.txt".format(i+1, j+1)).to_numpy()
            dataset_correct = np.vstack((dataset_correct,dataset_correct_))

correct_labels = np.ones((dataset_correct.shape[0],1))
dataset_correct = np.append(dataset_correct, correct_labels, axis = 1)

#read all file for incorrect movement
dataset_incorrect = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s01_e01_positions_inc.txt").to_numpy()
for i in range(0,10):
    for j in range(1,10):
        if i>=9 and j<9:
            dataset_incorrect_ = pd.read_csv("Incorrect Segmented Movements/Kinect/Positions/"+"m08_s{}_e0{}_positions_inc.txt".format(i+1, j+1)).to_numpy()
            dataset_incorrect = np.vstack((dataset_incorrect,dataset_incorrect_))
        elif i<9 and j>=9:
            dataset_incorrect_= pd.read_csv("Incorrect Segmented Movements/Kinect/Positions/"+"m08_s0{}_e{}_positions_inc.txt".format(i+1, j+1)).to_numpy()
            dataset_incorrect = np.vstack((dataset_incorrect,dataset_incorrect_))
        elif i>=9 and j>=9:
            dataset_incorrect_ = pd.read_csv("Incorrect Segmented Movements/Kinect/Positions/"+"m08_s{}_e{}_positions_inc.txt".format(i+1, j+1))
            dataset_incorrect = np.vstack((dataset_incorrect,dataset_incorrect_))
        else:
            dataset_incorrect_ = pd.read_csv("Incorrect Segmented Movements/Kinect/Positions/"+"m08_s0{}_e0{}_positions_inc.txt".format(i+1, j+1))
            dataset_incorrect = np.vstack((dataset_incorrect,dataset_incorrect_))

incorrect_labels = np.zeros((dataset_incorrect.shape[0],1))
dataset_incorrect = np.append(dataset_incorrect, incorrect_labels, axis = 1)

dataset = np.append(dataset_correct, dataset_incorrect, axis = 0)
savetxt('stroke_rehabilitation.csv', dataset, delimiter=',')
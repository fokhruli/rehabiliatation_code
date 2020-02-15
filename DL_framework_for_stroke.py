#importing libraries
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
from nltk import flatten
import matplotlib.pyplot as plt
import math
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

"""correct movement"""
#file reader
dataset = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s01_e01_positions.txt").to_numpy()
#dataset2 = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s02_e02_positions.txt").to_numpy()
#df = np.vstack((dataset1,dataset2))
for i in range(0,9):
    for j in range(1,9):
        if i<=8 and j<=8:
            dataset_ = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s0{}_e0{}_positions.txt".format(i+1,j+1)).to_numpy()
            dataset = np.vstack((dataset,dataset_))
        if i<=8 and j==9:
            dataset_ = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s0{}_e{}_positions.txt".format(i+1,j+1)).to_numpy()
            dataset = np.vstack((dataset,dataset_))
        if i==9 and j<=8:
            dataset_ = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s{}_e0{}_positions.txt".format(i+1,j+1)).to_numpy()
            dataset = np.vstack((dataset,dataset_))
        if i==9 and j==9:
            dataset_ = pd.read_csv("./Segmented Movements/Kinect/Positions/m08_s{}_e{}_positions.txt".format(i+1,j+1)).to_numpy()
            dataset = np.vstack((dataset,dataset_))
            

#making 20 frames into a row
frame_count = 33
#frame_skip = frame_count*(25*3)
frames =[]
for i in range(0,len(dataset)-frame_count,frame_count):
  frames.append(dataset[i:i+frame_count])
frames = np.array(frames)
  
"""incorrect movement"""
#file reader
dataset_inc = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s01_e01_positions_inc.txt").to_numpy()
for i in range(0,9):
    for j in range(1,9):
        if i<=8 and j<=8:
            dataset_ = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s0{}_e0{}_positions_inc.txt".format(i+1,j+1)).to_numpy()
            dataset_inc = np.vstack((dataset_inc,dataset_))
        if i<=8 and j==9:
            dataset_ = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s0{}_e{}_positions_inc.txt".format(i+1,j+1)).to_numpy()
            dataset_inc = np.vstack((dataset_inc,dataset_))
        if i==9 and j<=8:
            dataset_ = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s{}_e0{}_positions_inc.txt".format(i+1,j+1)).to_numpy()
            dataset_inc = np.vstack((dataset_inc,dataset_))
        if i==9 and j==9:
            dataset_ = pd.read_csv("./Incorrect Segmented Movements/Kinect/Positions/m08_s{}_e{}_positions_inc.txt".format(i+1,j+1)).to_numpy()
            dataset_inc = np.vstack((dataset_inc,dataset_))
        
#making 20 frames into a row
frames_inc =[]
for i in range(0,len(dataset_inc)-frame_count,frame_count):
  frames_inc.append(dataset_inc[i:i+frame_count])
frames_inc = np.array(frames_inc)

frames_total = np.vstack((frames,frames_inc))

"""performance matrix"""
#Using dtw distance between subject for refrence movement according to the paper
distances_ref = []
y = np.reshape(frames,(-1,frames.shape[2]))
for i in range(len(frames)):
    x = frames[i,:,:]
    distance, _ =  fastdtw(x, y, dist=euclidean)
    distances_ref.append(distance)
  
#Using dtw distance between subject for patient movement according to the paper
distances_pat = []
y = np.reshape(frames,(-1,frames.shape[2])) 
for i in range(len(frames_inc)):
   x = frames_inc[i,:,:]
   distance, _ =  fastdtw(x, y, dist=euclidean)
   distances_pat.append(distance)
   
    
"""scoring function using eqn 10 of the paper""" 
#calculate mean         
mean = 0
for i in range(len(distances_ref)):
    mean = mean + abs(distances_ref[i])
mean = (1/len(distances_ref))*mean

#calculate delta
delta = 0
for i in range(len(distances_ref)):
    delta = delta + (float(abs(distances_ref[i]))-mean)**2
delta = math.sqrt((1/len(distances_ref))*delta)

#calculate quality score
quality_score_refrence = []
quality_score_patient = []

for i in range(len(distances_ref)):
    quality_score = 1/(1+math.exp((distances_ref[i])/(mean+(3*delta))-3.2))
    quality_score_refrence.append(quality_score)
    
for i in range(len(distances_pat)):
    quality_score = 1/(1+math.exp((distances_ref[i])/((mean+(3*delta))-3.2))+((distances_pat[i]-distances_ref[i])/(3.2*(mean+(3*delta)))))
    quality_score_patient.append(quality_score)

"""storing the score into csv file"""
with open("refrence movement label.csv",'w') as f:
    file = csv.writer(f)
    file.writerow(quality_score_refrence)
with open("patient movement label.csv",'w') as f:
    file = csv.writer(f)
    file.writerow(quality_score_patient)

"""Making the dataset for neural network"""
X = frames_total
X = np.reshape(X,(-1,X.shape[2]))
#feature scaling
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

X = np.reshape(X,(frames_total.shape[0],frames_total.shape[1],frames_total.shape[2]))
y = np.append(np.array(quality_score_refrence),np.array(quality_score_patient),axis=0)

"""train test split"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""Making the neural network"""
#import tensorflow.kerasA

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.layers import LeakyReLU

model = Sequential()
model.add(Convolution1D(60, 5, padding ='same', strides = 2, input_shape = (X_train.shape[1],X_train.shape[2])))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Convolution1D(30, 3, padding ='same', strides = 2))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Convolution1D(10, 3, padding ='same'))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(200))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(100))
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer = tf.optimizers.Adam(lr=0.001), loss = 'mse')

history = model.fit(X_train, y_train, epochs = 2500, validation_split=0.2)

#validation loss vs training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#predicted value
y_pred = model.predict(X_test)

#Saving the architecture (topology) of the network
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

#Saving network weights
model.save_weights("model_weights.h5")

#ploting predicted value vs actual value
plt.plot(y_pred,'bo',y_test,'y*')
plt.title('rehabilitaion scores')
plt.xlabel('time')
plt.ylabel('scores')
plt.legend()
plt.show()

"""Loading the model"""
with open("model.json", 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# load weights into new model
model.load_weights("model_weights.h5")

"""getting the predicted label"""
#predicted value
y_pred = model.predict(X_test)

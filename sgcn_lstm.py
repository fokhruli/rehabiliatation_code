import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM, Lambda, UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error
np.random.seed(1337)  # for reproducibility

dataset = pd.read_csv("F:/final thesis/stroke rehabitation/code/rehabiliatation_code-master/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises-master/Neural Networks/Spatio-Temporal NN for Kinect v2/Data_KIMORE_e5/Train_X.csv", header = None)
Joint_Position = dataset.iloc[:,:].values
Joint_Position = Joint_Position.astype('float32')
zero = np.zeros((Joint_Position.shape[0],13)) 
Joint_Position = np.append(Joint_Position,zero,axis=1)

label = pd.read_csv('F:/final thesis/stroke rehabitation/code/rehabiliatation_code-master/A-Deep-Learning-Framework-for-Assessing-Physical-Rehabilitation-Exercises-master/Neural Networks/Spatio-Temporal NN for Kinect v2/Data_KIMORE_e5/Train_Y.csv', header = None)
y_train = label.iloc[:,:].values
y_train = np.reshape(y_train,(-1,1)).astype('float32') 


index_Spine_Base=0
index_Spine_Mid=4
index_Neck=8
index_Head=12   # no orientation
index_Shoulder_Left=16
index_Elbow_Left=20
index_Wrist_Left=24
index_Hand_Left=28
index_Shoulder_Right=32
index_Elbow_Right=36
index_Wrist_Right=40
index_Hand_Right=44
index_Hip_Left=48
index_Knee_Left=52
index_Ankle_Left=56
index_Foot_Left=60  # no orientation    
index_Hip_Right=64
index_Knee_Right=68
index_Ankle_Right=72
index_Foot_Right=76   # no orientation
index_Spine_Shoulder=80
index_Tip_Left=84     # no orientation
index_Thumb_Left=88   # no orientation
index_Tip_Right=92    # no orientation
index_Thumb_Right=96  # no orientation

body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left, index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right, index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left, index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Ankle_Right, index_Spine_Shoulder, index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
]

num_joints = len(body_parts)
num_timestep = 100
num_channel = 4
batch_size = Joint_Position.shape[0]//100

X_train = np.zeros((Joint_Position.shape[0],num_joints*num_channel)).astype('float32')

for row in range(Joint_Position.shape[0]):
    counter = 0
    for parts in body_parts:
        for i in range(4):
            X_train[row, counter+i] = Joint_Position[row, parts+i]
        counter += 4 

X_train_ = np.zeros((batch_size, num_timestep, num_joints, num_channel))

for batch in range(X_train_.shape[0]):
    for timestep in range(X_train_.shape[1]):
        for node in range(X_train_.shape[2]):
            for channel in range(X_train_.shape[3]):
                X_train_[batch,timestep,node,channel] = X_train[timestep+(batch*100),channel+(node*4)]
                
X_train = X_train_

    
# Split the data into training and validation sets
train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.3)


def normalize_adjacency(num_node):
    """
    Parameters
    ----------
    num_node : Int
        Total number of nodes(joints)

    Returns
    -------
    AD : Tensor
        Returns the normalized adjacency matrix
    """
    self_link = [(i, i) for i in range(num_node)]
    neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
    edge = self_link + neighbor_link    
    A = np.zeros((num_node, num_node)) # adjacency matrix
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    
    A2 = np.zeros((num_node, num_node)) # second order adjacency matrix
    for root in range(A.shape[1]):
        for neighbour in range(A.shape[0]):
            if A[root, neighbour] == 1:
                for neighbour_of_neigbour in range(A.shape[0]):
                    if A[neighbour, neighbour_of_neigbour] == 1:
                        A2[root,neighbour_of_neigbour] = 1
                        
    AD = normalize(A)
    AD2 = normalize(A2)
    return AD, AD2
    
def normalize(adjacency):
    """
    Parameters
    ----------
    adjacency : Numpy array
        Adjacency matrix with shape (number of nodes, number of nodes).

    Returns
    -------
    normalize_adjacency : Tensor
        The normalize adjacency matrix.
    """
    rowsum = np.array(adjacency.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    normalize_adjacency = r_mat_inv.dot(adjacency)
    normalize_adjacency = normalize_adjacency.astype('float32')
    normalize_adjacency = tf.convert_to_tensor(normalize_adjacency)   
    return normalize_adjacency

AD, AD2 = normalize_adjacency(num_joints)


def sgcn(input):
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=1, activation='relu')(input)
    #x = Dropout(0.2)(x)
    gcn_1 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD, x])
    gcn_2 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD2, x])
    x = concatenate([gcn_1, gcn_2], axis=-1)                                                                                                                                   
    
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu')(x)
    #x = Dropout(0.2)(x)
    gcn_1 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD, x])
    gcn_2 = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD2, x])
    x = concatenate([gcn_1, gcn_2], axis=-1)
    
    #x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1,1), strides=1, activation='relu')(x)
    #x = Dropout(0.2)(x)
    #x = tf.keras.layers.Lambda(lambda x: tf.einsum('vw,ntwc->ntvc', x[0], x[1]))([AD, x])
    
    x = tf.keras.layers.Reshape(target_shape=(-1,x.shape[2]*x.shape[3]))(x)
    return x    

input = Input(shape=(None, train_x.shape[2], train_x.shape[3]), batch_size=None)
x = sgcn(input)

rec = LSTM(80, return_sequences=True)(x)
rec = Dropout(0.2)(rec)
rec1 = LSTM(40, return_sequences=True)(rec)
rec1 = Dropout(0.2)(rec1)
rec2 = LSTM(40, return_sequences=True)(rec1)
rec2 = Dropout(0.2)(rec2)
rec3 = LSTM(80)(rec2)
rec3 = Dropout(0.2)(rec3)
out = Dense(1, activation = 'linear')(rec3)

model = Model(input, out)
model.compile(loss='mse', optimizer= tf.keras.optimizers.Adam(lr=0.0001))
history = model.fit(train_x, train_y, validation_data = (valid_x,valid_y), epochs=200, batch_size=10)

# Plot the results
plt.figure(1)
plt.plot(history.history['loss'], 'b', label = 'Training Loss')
plt.title('Training Loss')
plt.plot(history.history['val_loss'], 'r', label = 'Validation Loss')
plt.legend()
plt.tight_layout()
#plt.savefig('SpatioGraphConvolutionLSTM.png', dpi=300)
plt.show()

#prediction
y_pred = np.zeros((valid_x.shape[0],1))
for samples in range(valid_x.shape[0]):
    y_pred[samples,0] = model.predict(tf.expand_dims(valid_x[samples,:,:,:],axis=0))

y_pred = model.predict(valid_x)

#garbage_value = tf.random.normal(shape=(1,500,21,4))
#garbage_prediction = model.predict(garbage_value)

# ploting 
plt.figure(figsize = (8,8))
plt.subplot(2,1,1)
plt.plot(y_pred,'s', color='red', label='Prediction', linestyle='None', alpha = 0.5, markersize=6)
#plt.savefig('SpatioGraphConvolutionLSTM_Score.png', dpi=300)
plt.plot(valid_y,'o', color='green',label='Quality Score', alpha = 0.4, markersize=6)

# Calculate the cumulative deviation and rms deviation for the validation set
test_dev = abs(np.squeeze(y_pred)-valid_y)
# Cumulative deviation
mean_abs_dev = np.mean(test_dev)
# RMS deviation
rms_dev = sqrt(mean_squared_error(y_pred, valid_y))
print('Mean absolute deviation:', mean_abs_dev)
print('RMS deviation:', rms_dev)
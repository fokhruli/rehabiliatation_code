#import tensorflow as tf
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import Conv2D, MaxPool2D
#from tensorflow.keras.optimizers import Adam
#print(tf.__version__)
import pandas as pd
import numpy as np
from preprocessing import get_frames
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

Fs = 10
frame_size = Fs*2 # 20
hop_size = Fs*1 # 10

X_train, y_train = get_frames(frame_size, hop_size)

X_ = np.concatenate((X_train, y_train), axis =1)
#np.random.shuffle(X)

X = X_[:,:-1]
y = X_[:,-1]
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, -1:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN

# Importing the Keras libraries and packages
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense

#import tensorflow.kerasA
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 12, activation = 'relu', input_shape = (X_train.shape[1],)))
classifier.add(tf.keras.layers.Dropout(0.4))

# Adding the second hidden layer
classifier.add(Dense(units = 24, activation = 'relu'))
classifier.add(tf.keras.layers.Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = tf.optimizers.Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, epochs = 500, validation_split=0.2)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
for i in range(y_pred.shape[0]):
    if y_pred[i,0] > 0.5:
        y_pred[i,0] = 1
    else:
        y_pred[i,0] = 0

#confusion matrix and accuracy        
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])
print("Accuracy is: ",accuracy)
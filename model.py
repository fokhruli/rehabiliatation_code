#import tensorflow as tf
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import Conv2D, MaxPool2D
#from tensorflow.keras.optimizers import Adam
#print(tf.__version__)
import pandas as pd
import numpy as np
from preprocessing import get_data
X, y = get_data()


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
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 512, activation = 'relu', input_shape = (360,)))
# units = (input + output)/2
# Adding the second hidden layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 64, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'RMSProp', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
# Fitting the ANN to the Training set
a = classifier.fit(X_train, y_train, batch_size = 1, epochs = 100, validation_split=0.2)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

'''
## incomplete task may change input shape
def model():
    model = Sequential()
    model.add(Dense(units = 512, activation = 'relu', input_shape = X.shape))
    model.add(Dropout(0.1))
    
    model.add(Dense(units = 512, activation='relu'))
    model.add(Dropout(0.2))
    
    #model.add(Flatten())
    
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    return model

model = model()

model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 10, verbose=1)
'''

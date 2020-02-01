import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
from preprocessing import feature_extruction
X, y = feature_extruction()

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


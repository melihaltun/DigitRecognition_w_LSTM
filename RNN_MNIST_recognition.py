import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #CuDNNLSTM
from tensorflow.keras.optimizers import legacy as keras_legacy

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize inputs
x_train = x_train/255.0
x_test = x_test/255.0
x_test = x_test-np.mean(x_train)
x_train = x_train-np.mean(x_train)

print(x_train.shape)
print(x_test.shape)

model = Sequential()
model.add(LSTM(128, input_shape=(x_test.shape[1:]), activation='relu', return_sequences=True))
#model.add(CuDNNLSTM(128, input_shape=(x_test.shape[1:]), return_sequences=True))           # CuDNNLSTM is faster but requires an Nvidia graphics card
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
#model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

opt = keras_legacy.Adam(learning_rate=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=12, validation_data=(x_test, y_test))

dummy = 1

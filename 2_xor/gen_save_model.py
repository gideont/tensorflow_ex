import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

import version_check as vc
from datetime import datetime
from tensorflow import keras

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

vc.version_check()

# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# fix random seed for reproducibility
numpy.random.seed(7)
# load or dataset
dataset = numpy.loadtxt("xor_gate.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:2]
Y = dataset[:,2]

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# create model
model = Sequential()
model.add(Dense(12, input_dim=2, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(
        X,
        Y, 
        epochs=5000, 
        batch_size=10, 
        verbose=1,
        callbacks=[tensorboard_callback],
)

# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


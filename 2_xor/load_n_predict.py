# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import tensorflow as tf
import numpy
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from numpy import array

print("\nThis is an example of XOR using Tensorflow Keras")
print("================================================")
print("TF version is: %s" % tf.VERSION)
print("Keras version is: %s" % tf.keras.__version__)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

Z = array([[0,1], [0,0], [1,1], [1,0]])
print("\nThe test array is:")
print(Z)

predictions = loaded_model.predict(Z)
# round predictions
rounded = [round(x[0]) for x in predictions]
print("\nThe result is: %s " % rounded)

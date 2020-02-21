import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

import tensorflow as tf
from tensorflow.keras import layers

print("TF version is: %s" % tf.__version__)
print("Keras version is: %s" % tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


print("Build with CUDA: %s" % tf.test.is_built_with_cuda())


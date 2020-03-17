import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'}

# set CPU=1 if want to use CPU only for tensorflow
CPU=0

if(CPU == 1):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # or even "-1"

import sys

# Confirm that we're using Python 3
assert sys.version_info.major is 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'

import tensorflow as tf
from tensorflow.keras import layers

def version_check():
    print("TF version: {}".format(tf.__version__))
    print("Keras version: {}".format(tf.keras.__version__))
    print("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    print("Build with CUDA: {}".format(tf.test.is_built_with_cuda()))
    if tf.test.gpu_device_name():
      print('GPU found')
    else:
      print("No GPU found")

if __name__== "__main__":
  version_check()




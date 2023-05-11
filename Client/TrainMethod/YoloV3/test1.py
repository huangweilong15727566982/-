# from tensorflow.python.client import device_lib
 
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     print(local_device_protos)
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# get_available_gpus()
# import tensorflow as tf




# tf.config.list_physical_devices('GPU')
import tensorflow as tf
import sys
print("TensorFlow版本：", tf.__version__)
sys.path.append("") 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Using GPU:', tf.test.gpu_device_name())
else:
    print('No GPU available')

# model = tf.keras.Sequential( ... )
# model.compile( ... )
# model.fit( ... , use_multiprocessing=True, workers=4)
import tensorflow as tf

# choose a device type such as CPU or GPU
devices = tf.config.list_physical_devices('GPU')
print(devices[0])

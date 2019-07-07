import tensorflow as tf
import ipdb

def build_model(input_shape, nb_output):
    model = tf.keras.models.Sequential()
    inputs = tf.placeholder(dtype=tf.float32, shape = [None,]+input_shape, name="input")
    model.add(tf.keras.layers.Dense(16, activation="relu", input_shape =[None,]+input_shape))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(nb_output))
    outputs = model(inputs)
    max_outputs = tf.reduce_max(outputs, reduction_indices=1)

    return inputs, outputs, max_outputs, model

import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    """
    Keras layer implementing the Attention mechanism.
    Used to help the model focus on important parts of the input sequence.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        et = tf.squeeze(tf.tanh(tf.matmul(x, self.W) + self.b), axis=-1)
        at = tf.nn.softmax(et)
        at = tf.expand_dims(at, axis=-1)
        output = x * at
        return tf.reduce_sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()

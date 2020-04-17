import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, huggingface_token_classification):
        super(Model, self).__init__()
        self.huggingface_token_classification = huggingface_token_classification
        
    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        return self.huggingface_token_classification(inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
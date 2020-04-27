import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForTokenClassification
from model.model import Model

''' parameters '''
model_name = 'bert-base-cased'
num_labels = 2
max_length = 64
weights_path = 'model/saved_weights/weights.0.21.h5'

''' load model '''
tokenizer = BertTokenizer.from_pretrained(model_name)
model = Model(TFBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)) # need to optimize this step by loading config instead of weights
model(tf.zeros([1, 3, max_length], tf.int32))
model.load_weights(weights_path)
model.compile(run_eagerly=True)

''' score passages '''
TEXT = "The origin of the name Moabit is disputed. According to one account, \
it can be traced back to the Huguenots, in the time of King Frederick William I of Prussia. \
These French refugees are said to have named their new residence in reference \
to the Biblical description of the Israelites in the country of Moab, where they \
stayed before being allowed to enter Canaan. Other possible origins include \
the German (Berlin dialect) \"Moorjebiet\" (swamp area). "

inputs_ = tokenizer.encode_plus(text=TEXT,
                                max_length=max_length,
                                pad_to_max_length=True,
                                return_token_type_ids=True, 
                                return_attention_mask=True)

inputs = tf.constant([[inputs_['input_ids'],
                inputs_['attention_mask'],
                inputs_['token_type_ids']
        ]])

output = model(inputs)
output = tf.math.softmax(output, axis=-1)
result = tf.math.argmax(output, axis=-1).numpy()
toks = tokenizer.convert_ids_to_tokens(inputs_['input_ids'], True)

for tok, label in zip(toks, result[0][0][:len(toks)]):
    print('{} - {}'.format(tok, label))

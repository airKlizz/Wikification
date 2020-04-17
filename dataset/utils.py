from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

def pad_to_max_length(labels, max_length):
    return ([0] + labels + [0] * max(0, max_length-len(labels)-1))[:max_length]

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, shuffle=10000, random_state=2020):

    with open(train_path, 'r') as f:
        data = f.read()

    passages = data.split('\n\n')

    X = []
    y = []
    for passage in passages:
        text = ""
        y_ = []
        lines = passage.split('\n')
        for line in lines:
            elems = line.split('\t')
            assert len(elems) == 2, elems
            string = elems[0]
            label = int(elems[1])
            toks = tokenizer.encode(string, add_special_tokens=False)
            text += string
            y_ += [label]*len(toks)
        
        inputs = tokenizer.encode_plus(text=text,
                                        max_length=max_length,
                                        pad_to_max_length=True,
                                        return_token_type_ids=True, 
                                        return_attention_mask=True)

        X.append([inputs['input_ids'],
                inputs['attention_mask'],
                inputs['token_type_ids']
        ])
        y.append(pad_to_max_length(y_, max_length))

    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset



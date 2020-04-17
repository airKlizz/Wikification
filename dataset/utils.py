from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def split_passage_per_link(passage):
    assert passage[-1] != '>', 'The split passage must be impair'
    split_passage = [string_2 for string_1 in passage.split('<a>') for string_2 in string_1.split('</a>')]
    if passage[0] == '<': pattern = [0, 1]
    else: pattern = [1, 0]
    labels = [pattern[1]]
    for _ in range(int((len(split_passage)-1)/2)):
        labels += pattern
    return split_passage, labels

def pad_to_max_length(labels, max_length):
    return ([0] + labels + [0] * max(0, max_length-len(labels)-1))[:max_length]

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_articles, shuffle=10000, random_state=2020):

    with open(train_path, 'r') as f:
        data = f.read()

    articles = data.split('\n\n')

    X = []
    y = []
    for article in tqdm(articles[:num_articles], desc='Data processing...'):
        passages = article.split('\n')[1:]
        for passage in passages:
            text = ""
            y_ = []
            try:
                strings, labels = split_passage_per_link(passage)
                assert len(strings) == len(labels)
            except:
                print('ERROR: The split passage must be impair')
                continue

            for string, label in zip(strings, labels):
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
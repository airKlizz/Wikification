import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForTokenClassification
from dataset.utils import create_tf_dataset
from model.model import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="Name of the HugginFace Model", default="bert-base-cased")
parser.add_argument("--train_path", type=str, help="path to the train  file", default="data/data.txt")
parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=256)
parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
parser.add_argument("--batch_size", type=int, help="batch size", default=32)
parser.add_argument("--num_articles", type=int, help="num of articles to train on", default=1000)
parser.add_argument("--num_labels", type=int, help="number of labels", default=2)
parser.add_argument("--epochs", type=int, help="number of epochs", default=5)
parser.add_argument("--learning_rate", type=float, help="learning rate", default=3e-5)
parser.add_argument("--filepath", type=str, help="filename for saving", default="model/saved_weights/weights.{epoch:02d}-{val_loss:.2f}.h5")
args = parser.parse_args()

# parameters
model_name = args.model_name
num_labels = args.num_labels
train_path = args.train_path
max_length = args.max_length
test_size = args.test_size
batch_size = args.batch_size
num_articles = args.num_articles
learning_rate = args.learning_rate
epochs = args.epochs
filepath = args.filepath

# init tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = Model(TFBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels))

# create dataset
train_dataset, validation_dataset = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_articles)

# optimizer, loss and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# compile
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# callbacks
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)


# train
history = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=[model_checkpoint])

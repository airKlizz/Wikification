# Wikification

This objective is to retrieve wikify a text using the token classification Bert model from Hugging Face. 

## Dataset

I create a dataset composed of wikipedia passages with hyperlinks (*words between \<a\> and \</a\>*) to others wikipedia articles. 

Example of a passage :

```
Patrick Lavon Mahomes II (born September 17, 1995) is an  <a>American football</a> <a>quarterback</a> for the <a>Kansas City Chiefs</a> of the <a>National Football League</a> (NFL). He is the son of former <a>Major League Baseball</a> (MLB) pitcher <a>Pat Mahomes</a>. He initially played <a>college football</a> and college <a>baseball</a> at <a>Texas Tech University</a>. Following his sophomore year, he quit baseball to focus solely on football. In his junior year, he led all <a>NCAA Division I FBS</a> players in multiple categories including passing yards (5,052 yards) and passing touchdowns (53 touchdowns). He then entered the <a>2017 NFL Draft</a> where he was the tenth overall selection by the Kansas City Chiefs.
```

Data files are available [here](https://drive.google.com/open?id=14CYMrUx3rQk0E17W_lKzdRgp0AHM6xp1). Then ``.txt`` files jave to be in ``data/`` folder.

The train data are composed of the [top 5000 of wikipedia articles of 2019](https://en.wikipedia.org/wiki/User:West.andrew.g/2019_Popular_pages) and around 5000 other random articles. The test data are composed of 1000 random articles. 

## Getting started

### Training

You need to have downloaded the data files and put them on the ``data/`` folder.

```
python train.py --train_path data/data.txt \
                  --max_length 256 \
                  --batch_size 16 \
                  --num_articles 10000 \
                  --num_labels 2 \
                  --epochs 4 \
                  --learning_rate 1e-6 \
                  --filepath model/saved_weights/weights.{epoch:02d}-{val_loss:.2f}.h5 \
                  --test_data_path data/test.data.txt \
                  --test_gold_path data/test.gold.txt \
                  --candidate_path data/candidate.txt \
                  --weight_for_0 0.9 \
                  --weight_for_1 50
```

To see full usage of ``train.py``, run ``python train.py --help``.

### Evaluation

You need to have downloaded the model weights from [here](https://github.com/airKlizz/Wikification/releases/download/v1.0-model_weights/weights.loss.0.21.h5).

```python

''' imports '''
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForTokenClassification
from model.model import Model
from evaluation.utils import run_evaluation, predict_passage, get_entities_from_passage

''' parameters '''
model_name = 'bert-base-cased'
num_labels = 2
max_length = 64
weights_path = 'path/to/weights/file.h5'

''' load model '''
tokenizer = BertTokenizer.from_pretrained(model_name)
model = Model(TFBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)) # need to optimize this step by loading config instead of weights
model(tf.zeros([1, 3, max_length], tf.int32))
model.load_weights(weights_path)
model.compile(run_eagerly=True)

''' eval parameters '''
batch_size = 16
test_data_path = 'data/test.data.txt'
test_gold_path = 'data/test.gold.txt'
candidate_path = 'data/candidate.txt'

metrics = run_evaluation(model, tokenizer, max_length, batch_size, test_data_path, test_gold_path, candidate_path)
for metric, value in metrics.items():
  print('{} - \t{:.3f}'.format(metric, value))
  
# recall - 	0.271
# precision - 	0.282
# f1 - 	0.276
```

The metrics are low but the fact that the task is quite subjective may explain this. I did not find other models to compare with my results. If you know, I am interested.

### Use pre-trained models

```python

''' imports '''
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForTokenClassification
from model.model import Model
from evaluation.utils import run_evaluation, predict_passage, get_entities_from_passage
from mediawiki.utils import get_articles

''' parameters '''
model_name = 'bert-base-cased'
num_labels = 2
max_length = 64
weights_path = 'path/to/weights/file.h5'

''' load model '''
tokenizer = BertTokenizer.from_pretrained(model_name)
model = Model(TFBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)) # need to optimize this step by loading config instead of weights
model(tf.zeros([1, 3, max_length], tf.int32))
model.load_weights(weights_path)
model.compile(run_eagerly=True)

TEXT_TO_WIKIFY = "A Huguenot and officer under Admiral Gaspard de Coligny, \
Ribault led an expedition to the New World in 1562 that founded the outpost of Charlesfort \
on Parris Island in present-day South Carolina."

text_wikified = predict_passage(TEXT_TO_WIKIFY, model, tokenizer, max_length)
entities = get_entities_from_passage(text_wikified)
articles = get_articles(entities)

print(text_wikified)
# A <a>Huguenot</a> and officer under <a>Admiral Gaspard de Coligny</a> , 
# <a>Ribault</a> led an expedition to the <a>New World</a> in <a>1562</a> that founded the <a>outpost</a> of <a>Charlesfort</a> 
# on <a>Parris Island</a> in present <a>-</a> day <a>South Carolina</a> .

print(entities)
#['Huguenot', 'Admiral Gaspard de Coligny', 'Ribault', 'New World', '1562', 'outpost', 'Charlesfort', 'Parris Island', '-', 'South Carolina']

print([article['title'] for article in articles])
#['Huguenots', 'Gaspard II de Coligny', 'Jean Ribault', 'New World', '1562', 'Outpost', 'Charlesfort-Santa Elena Site', 'Marine Corps Recruit Depot Parris Island', 'Apostrophe', 'South Carolina']
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

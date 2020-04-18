import numpy as np
import tensorflow as tf
from tqdm import tqdm

def convert_tokens_to_offsets(tokens):
  offsets = []
  last_idx = 0
  for idx in range(1, len(tokens)+1):
    if idx == len(tokens):
      offsets.append((last_idx, idx))
    else:
      if tokens[idx][:2] != '##':
          offsets.append((last_idx, idx))
          last_idx = idx
  return offsets

def annote_passage(passage, result, tokenizer):

  tokens = tokenizer.tokenize(passage)
  offsets = convert_tokens_to_offsets(tokens)
  result = np.array(result[1:len(tokens)+1])

  entity_begin = tokenizer.tokenize('<a>')
  entity_end = tokenizer.tokenize('</a>')

  new_tokens = []
  for begin, end in offsets:
    if end > len(result): # no result after the index
      entity = False
    else:
      entity = True if np.sum(result[range(begin, end)]) > 0 else False
    if entity: new_tokens += entity_begin
    for i in range(begin, end):
      new_tokens.append(tokens[i])
    if entity: new_tokens += entity_end
  
  new_passage = tokenizer.convert_tokens_to_string(new_tokens)
  new_passage = new_passage.replace('< a > ', '<a>')
  new_passage = new_passage.replace(' < / a >', '</a>')
  new_passage = new_passage.replace('</a> <a>', ' ')
  return new_passage

def predict_passage(passage, model, tokenizer, max_length):
  inputs_ = tokenizer.encode_plus(text=passage,
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
  result = result[0][0]

  return annote_passage(passage, result, tokenizer)

def predict_passages(passages, model, tokenizer, max_length, batch_size):
  if passages == "": return passages
  batch_inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=passages,
                                max_length=max_length,
                                pad_to_max_length=True,
                                return_token_type_ids=True, 
                                return_attention_mask=True)
  print(batch_inputs, passages)
  batch_ids = tf.expand_dims(batch_inputs['input_ids'], 1)
  batch_attention_mask = tf.expand_dims(batch_inputs['attention_mask'], 1)
  batch_type_ids = tf.expand_dims(batch_inputs['token_type_ids'], 1)

  inputs = tf.concat([batch_ids, batch_attention_mask, batch_type_ids], axis=1)

  outputs = model.predict(inputs, batch_size=batch_size)
  outputs = tf.math.softmax(outputs, axis=-1)
  results = tf.math.argmax(outputs, axis=-1).numpy()
  results = results[0]

  new_passages = []
  for passage, result in zip(passages, results):
    new_passages.append(annote_passage(passage, result, tokenizer))
  
  return new_passages

def get_entities_from_passage(passage):
  entities = []
  split_passage = [string_2 for string_1 in passage.split('<a>') for string_2 in string_1.split('</a>')]
  for i in range(1, len(split_passage)-1, 2):
    entities.append(split_passage[i])
  return entities

def get_entities_from_passages(passages):
  entities = []
  for passage in passages:
    entities.append(get_entities_from_passage(passage))
  return entities

def create_candidate(model, tokenizer, max_length, batch_size, test_data_path, candidate_path):
  with open(test_data_path, 'r') as f:
    data = f.read()
  articles = data.split('\n\n')
  with open(candidate_path, 'w') as f:
    for article in tqdm(articles):
      passages = article.split('\n')
      f.write(passages[0])
      f.write('\n')
      new_passages = predict_passages(passages[1:], model, tokenizer, max_length, batch_size)
      for passage in new_passages:
        f.write(passage)
        f.write('\n')
      f.write('\n')

def get_passages_from_file(filename):
  with open(filename, 'r') as f:
    data = f.read()
  passages = []
  for article in data.split('\n\n'):
    passages += article.split('\n')[1:]
  return passages

def get_metrics(candidate_path, gold_path):
  candidate_passages = get_passages_from_file(candidate_path)
  gold_passages = get_passages_from_file(gold_path)

  all_candidate_entities = get_entities_from_passages(candidate_passages)
  all_gold_entities = get_entities_from_passages(gold_passages)

  recall = 0
  precision = 0
  num_gold_entities = 0
  num_candidate_entities = 0

  for candidate_entities, gold_entities in zip(all_candidate_entities, all_gold_entities):
    
    for ent in gold_entities:
      if ent in ' '.join(candidate_entities):
        recall += 1
    num_gold_entities += len(gold_entities)

    for ent in candidate_entities:
      if ent in ' '.join(gold_entities):
        precision += 1
    num_candidate_entities += len(candidate_entities)

  recall /= num_gold_entities
  precision /= num_candidate_entities

  return {
      'recall': recall,
      'precision': precision,
      'f1': 2*((precision*recall)/(precision+recall))
  }

def run_evaluation(model, tokenizer, max_length, batch_size, test_data_path, test_gold_path, candidate_path):
  create_candidate(model, tokenizer, max_length, batch_size, test_data_path, candidate_path)
  return get_metrics(candidate_path, test_gold_path)

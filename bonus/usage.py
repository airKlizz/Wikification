import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForTokenClassification
from model.model import Model
from evaluation.utils import run_evaluation, predict_passage, get_entities_from_passage
from .mediawiki import get_articles

def get_entities_from_text(text, model, tokenizer, max_length):
    text_wikified = predict_passage(text, model, tokenizer, max_length)
    entities = get_entities_from_passage(text_wikified)
    return entities





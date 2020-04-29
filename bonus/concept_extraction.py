from .usage import *

from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from tqdm import tqdm


def get_wikification(scores_passages, model, tokenizer, max_length):
    scores = []
    passages = []
    wiki_entities = []

    for score, passage in tqdm(scores_passages.items()):
        scores.append(score)
        passages.append(passage)
        entities = get_entities_from_text(passage, model, tokenizer, max_length)
        clean_entities = []
        for entity in entities:
            if len(entity) < 3: continue 
            clean_entities.append(entity)

        wiki_entities.append(clean_entities)

    wikification = list(zip(passages, scores, wiki_entities))
    return wikification

def get_entities_best_passages(wikification):
    entities_best_passages = {}
    for passage, _, wiki_entities in sorted(wikification, key=lambda wikification: wikification[1]):
        for entity in wiki_entities:
            entities_best_passages[entity] = passage
            sentences = sent_tokenize(passage)
            for sentence in sentences:
                if entity in sentence:
                    entities_best_passages[entity] = sentence
    return entities_best_passages

def score_wiki_entities(wikification):
    wiki_entities_scores = {}
    for _, score, wiki_entities in wikification:
        for entity in wiki_entities:
            if entity in wiki_entities_scores.keys():
                wiki_entities_scores[entity] += score
            else:
                wiki_entities_scores[entity] = score
    return wiki_entities_scores

def clustering_wiki_entities(wiki_entities_scores, distance_threshold=1):
    embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1")
    wiki_entities = list(wiki_entities_scores.keys())
    embeddings = embed(wiki_entities)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit(embeddings)

    num_clusters = max(clustering.labels_)+1
    clusters = []
    for i in range(num_clusters):
        cluster_idxs = [j for j, label in enumerate(clustering.labels_) if label == i]
        cluster_wiki_entities = [wiki_entities[j] for j in cluster_idxs]
        cluster_sum_score = sum([wiki_entities_scores[entity] for entity in cluster_wiki_entities])
        cluster_embeddings = [embeddings[j] for j in cluster_idxs]
        cluster_mean = np.mean(cluster_embeddings, axis=0)
        cluter_representative_idx = np.argmax(cosine_similarity(cluster_embeddings, [cluster_mean]))
        cluster_representative = cluster_wiki_entities[cluter_representative_idx]
        cluster_top_entities_idx = np.argmax([wiki_entities_scores[entity] for entity in cluster_wiki_entities])
        cluster_top_entities = cluster_wiki_entities[cluster_top_entities_idx]
        
        cluster = (cluster_sum_score, cluster_representative, cluster_top_entities, cluster_wiki_entities)
        clusters.append(cluster)
    return clusters

def get_top_entities_scores(clusters):
    top_entities_scores = {}
    for cluster_sum_score, _, cluster_top_entities, _ in sorted(clusters, key=lambda clusters: clusters[0], reverse=True):
        top_entities_scores[cluster_top_entities] = cluster_sum_score
    return top_entities_scores

def words_similarity(topic, entity):
    porter = PorterStemmer()
    topic_words = [porter.stem(word) for word in topic.split(' ')]
    entity_words = [porter.stem(word) for word in entity.split(' ')]
    for word in topic_words:
        if word in entity_words:
            return True
    return False

def concept_extraction(topic, scores_passages, model, tokenizer, max_length, top_n=10, exclude_topic_words=True, remove_short=True, distance_threshold=1):
    wikification  = get_wikification(scores_passages, model, tokenizer, max_length)
    entities_best_passages = get_entities_best_passages(wikification)
    wiki_entities_scores = score_wiki_entities(wikification)
    clusters = clustering_wiki_entities(wiki_entities_scores, distance_threshold=distance_threshold)
    top_entities_scores = get_top_entities_scores(clusters)
    top_entities = sorted(top_entities_scores, key=top_entities_scores.get, reverse=True)
    if exclude_topic_words:
        top_entities = [top_entity for top_entity in top_entities if words_similarity(topic, top_entity) == False]
    if remove_short:
        top_entities = [top_entity for top_entity in top_entities if len(top_entity) > 3]
    return top_entities[:top_n] 

def filename_to_scores_passages(filename):
    with open(filename, 'r') as f:
        data = f.read()
    scores_passages = {}
    for line in data.split('\n'):
        elems = line.split('\t')
        scores_passages[float(elems[0])] = elems[1]
    return scores_passages
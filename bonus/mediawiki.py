import json
import requests

def get_article(entity):
    url_template = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&utf8=&format=json&srlimit=1"
    entity = entity.replace(' ', '_')
    url = url_template.format(entity)
    try:
        results = requests.get(url).json()
    except:
        return None
    results = results['query']['search']
    if len(results) == 0: return None
    return {'id': results[0]['pageid'], 'title': results[0]['title']}

def get_articles(entities):
    articles = []
    for entity in entities:
        article = get_article(entity)
        if article == None: continue
        articles.append(article)
    return articles
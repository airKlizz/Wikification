from requests import get
from bs4.element import Tag, NavigableString, Comment
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

def get_html(url):
    return get(url).content.decode("utf-8") 

def get_raw_passages(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.findAll('p')

def _remove_reference(string):
    return re.sub(r'\[[0-9]+\]', '', string)

def process_passage(passage):
    processed_passage = ""
    for content in passage.contents:
        if type(content) == NavigableString:
            processed_passage += str(content)
        elif type(content) == Tag:
            if content.name == 'a':
                processed_passage += '<a>{}</a>'.format(content.text)
            else:
                processed_passage += str(content.text)
        elif type(content) == Comment:
            processed_passage += str(content)
        else:
            assert False, 'Content type not known: '+str(type(content))
    return _remove_reference(processed_passage)

def split_passage_per_link(passage):
    assert passage[-1] != '>', 'The split passage must be impair'
    split_passage = [string_2 for string_1 in passage.split('<a>') for string_2 in string_1.split('</a>')]
    if passage[0] == '<': pattern = [0, 1]
    else: pattern = [1, 0]
    labels = [pattern[1]]
    for _ in range(int((len(split_passage)-1)/2)):
        labels += pattern
    return split_passage, labels

def add_article(url, filename):
    try:
        html = get_html(url)
    except:
        print('Request impossible to the url: {}'.format(url))
        return
    passages = get_raw_passages(html)
    with open(filename, 'a') as f:
        for passage in passages:
            passage = process_passage(passage)
            try:
                split_passage, labels = split_passage_per_link(passage)
            except:
                print('ERROR: The split passage must be impair')
                continue
            for p, l in zip(split_passage, labels):
                f.write('{}\t{}\n'.format(p.replace('\n', ' '), l))
            f.write('\n')

def get_top5000(filename='top5000_wikipedia'):
    with open(filename, 'r') as f:
        data = f.read()

    lines = data.split('\n')
    top5000 = {}
    for line in lines[2:]:
        elems = line.split('\t')
        rank = elems[0]
        rank = rank.replace(' ', '')
        rank = int(rank)
        title = elems[1][:-1]
        link = 'https://en.wikipedia.org/wiki/'+title.replace(' ', '_')
        top5000[rank] = link

    return top5000

def create_data(filename, top5000_filename='top5000_wikipedia'):
    top5000 = get_top5000(top5000_filename)
    for i, (_, url) in tqdm(enumerate(top5000.items())):
        add_article(url, filename)
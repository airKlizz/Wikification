from requests import get
from bs4.element import Tag, NavigableString, Comment
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import random

def get_html(url):
    return get(url).content.decode("utf-8") 

def get_raw_passages(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.findAll('p')

def _clean_passage(passage):
    passage = re.sub(r'\[[0-9]+\]', '', passage)
    passage = passage.replace('\n', '')
    passage = passage.replace('\t', ' ')
    return passage

def process_passage(passage):
    processed_passage = ""
    for content in passage.contents:
        if type(content) == NavigableString:
            processed_passage += str(content)
        elif type(content) == Tag:
            if content.name == 'a':
                if 'href' in content.attrs.keys():
                    if '/wiki/' in content.attrs["href"]:
                        processed_passage += '<a>{}</a>'.format(content.text)
                    else:
                        processed_passage += str(content.text)
                else:
                    processed_passage += str(content.text)
            else:
                processed_passage += str(content.text)
        elif type(content) == Comment:
            processed_passage += str(content)
        else:
            assert False, 'Content type not known: '+str(type(content))
    return _clean_passage(processed_passage)

def add_split_passage(url, filename):
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

def add_article(url, filename):
    try:
        html = get_html(url)
    except:
        print('Request impossible to the url: {}'.format(url))
        return
    with open(filename, 'a') as f:
        f.write(url)
        f.write('\n')
        passages = get_raw_passages(html)
        for passage in passages:
            passage = process_passage(passage)
            if len(passage) < 64: continue
            f.write(passage)
            f.write('\n')
        f.write('\n')

def get_top5000(filename='top5000_wikipedia'):
    with open(filename, 'r') as f:
        data = f.read()

    lines = data.split('\n')
    titles = []
    for line in lines[2:]:
        elems = line.split('\t')
        title = elems[1][:-1]
        title = title.replace(' ', '_')
        titles.append(title)

    return titles

def extract_good_titles(titles_file, new_titles_file):

    with open(titles_file, 'r') as f:
        with open(new_titles_file, 'w') as new_f:
            for i, line in enumerate(f):
                if i == 0: continue
                title = line[:-1].split('\t')[1]

                ''' Rules to determine what is a good title '''
                if not ('A' <= title[0] <= 'Z' or 'a' <= title[0] <= 'z'): continue
                if 'Wiki' in title: continue
                if len(title) < 5: continue
                if len(title) > 20: continue
                if re.match(r'.*[-!\.@#$%^&*:;()+=\\/\'"\"?><,~`\{\}\[\]].*', title): continue
                if re.match(r'.*_.{1, 3}_.*', title): continue
                if re.match(r'.*([a-zA-Z]+[0-9]+)|([0-9]+[a-zA-Z]+).*', title): continue

                new_f.write(title)
                new_f.write('\n')

def pick_titles(titles_file, num_titles):
    with open(titles_file, 'r') as f:
        data = f.read()
    titles = data.split('\n')
    random.shuffle(titles)
    return titles[:num_titles]

def get_titles(num_titles, top5000_filename='top5000_wikipedia', titles_filename='titles.wikipedia'):
    titles = get_top5000(top5000_filename)
    if len(titles) >= num_titles:
        return titles[:num_titles]
    titles += pick_titles(titles_filename, num_titles-len(titles))
    return titles

def create_data(num_titles, filename, top5000_filename='top5000_wikipedia', titles_filename='titles.wikipedia'):
    titles = get_titles(num_titles, top5000_filename, titles_filename)
    for title in tqdm(titles):
        add_article('https://en.wikipedia.org/wiki/{}'.format(title), filename)
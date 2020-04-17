import sys
from utils import add_article

COMMAND = "Command line: python add_article.py <url of the article> <data filename>"

def main():
    assert len(sys.argv) == 3, COMMAND
    filename = sys.argv[2] 
    url = sys.argv[1]
    add_article(url, filename)

if __name__ == '__main__':
    main()
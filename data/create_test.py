import sys
from utils import create_test

COMMAND = "Must have 3 arguments. You gave {} arguments. Command line: python create_test.py <number of articles> <test data filename> <test gold filename>"

def main():
    assert len(sys.argv) == 4, COMMAND.format(len(sys.argv)-1)
    num_articles = int(sys.argv[1]) 
    test_data_filename = sys.argv[2]
    test_gold_filename = sys.argv[3]
    print('Number of articles: {}'.format(num_articles))
    create_test(num_articles, test_data_filename, test_gold_filename)

if __name__ == '__main__':
    main()
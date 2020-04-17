import sys
from utils import create_data

COMMAND = "Command line: python create_data.py <number of articles> <data filename> <top5000 filename> <titles filename>"

def main():
    assert len(sys.argv) == 5, COMMAND
    num_titles = sys.argv[1]
    filename = sys.argv[2]
    top5000_filename = sys.argv[3] 
    titles_filename = sys.argv[4]
    
    create_data(num_titles, filename, top5000_filename, titles_filename)

if __name__ == '__main__':
    main()
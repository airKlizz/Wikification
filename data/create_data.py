import sys
from utils import create_data

COMMAND = "Command line: python create_data.py <data filename> <top5000 filename"

def main():
    assert len(sys.argv) == 3, COMMAND
    top5000_filename = sys.argv[2] 
    filename = sys.argv[1]
    create_data(filename, top5000_filename)

if __name__ == '__main__':
    main()
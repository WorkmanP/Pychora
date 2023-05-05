import argparse

parser = argparse.ArgumentParser()
parser.add_argument('tests', nargs='*', help='Test Args')
print(parser.parse_args())

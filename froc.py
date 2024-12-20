import json


INPUT_PATH = 'input/'
OUTPUT_PATH = 'output/'

with open(INPUT_PATH + 'lymphocytes.json', 'r') as f:
    lymphocytes = json.load(f)
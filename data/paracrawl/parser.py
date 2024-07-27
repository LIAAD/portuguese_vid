from pathlib import Path
import os

CURRENT_PATH = Path(__file__).parent

with open(os.path.join(CURRENT_PATH, 'data','en-pt.txt'), 'r', encoding='utf-8') as f:
    line = f.readline()
    print(line)
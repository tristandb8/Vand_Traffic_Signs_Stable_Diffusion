# Intermediate script used to create the dataset used to train
# the stable diffusion model

import os
from os import path
import json
import shutil
from PIL import Image

with open('clean/img_info.json', 'r') as inp:
    data = json.load(inp)

with open('signs-set/train/metadata.jsonl', 'w+') as out:
    for key, value in data.items():
        if value['train']:
            try:
                Image.open(path.join('clean', key))
                shutil.copyfile(path.join('clean', key), path.join('signs-set/train', key))
                out.write(json.dumps({'file_name': key, 'caption': 'vandalized street sign' + (f'; {value["alt"]}' if "alt" in value else "")}) + "\n")
            except Exception as e:
                print(e)

import os

path_str = '/data/ai-bandits/datasets/charades/data'

box = [0, 0, 1, 1]

delta = 0.005

all_classes = {'sit': 0, 'stand': 1, 'lie': 2, 'walk': 3, 'fall': 4, 'eat': 5, 'drink': 6, 'phone/camera': 7, 'book': 8,
               'television': 9, 'cook': 10, 'laptop': 11}

prev_lables = list(range(0, 13))
prev_uid = 'NONE'

datasets = ['train', 'test']

CHAR_Videos = os.path.join(path_str, "videos")
import datetime
from collections import defaultdict
import numpy as np

import math
import cv2 


def get_img_name() -> str:
    now = datetime.datetime.now()
    img_name = datetime.datetime.strftime(now, '%Y-%m-%d-%H-%M-%S') + '.jpg'
    return img_name


# num2name = defaultdict()
# num_points = list(range(18))
# name_points = ['nose', 'neck', 'lshoulder', 'lelbow', 'lhand', 'rshoulder',
#                 'relbow', 'rhand', 'lhip', 'lknee', 'lfoot', 'rhip', 'rknee',
#                 'rfoot', 'leye', 'reye', 'lear', 'rear']
# for i in range(18):
#     num2name[num_points[i]] = name_points[i]

# convert candidate (openpose result, [x, y, confidence, body part index]) into a dictionary
def dict_index(candidate):
    ans = defaultdict()
    for row in candidate: # candidate is a list of list
        key = int(row[-1])
        x = row[0]
        y = row[1]
        ans[key] = (x, y)
    return ans



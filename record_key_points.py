import sqlite3
from sqlite3 import Error
from pathlib import Path
import os
from loguru import logger




import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

import timeit
import argparse
from datetime import datetime
import time

from op import model
from op import util
from op.body import Body
from op.hand import Hand


from utils import *
from detect_position.position_detection import baseline


body_estimation = Body('body_pose_model.pth')

model = baseline(36, 20, 10, 8, 5, 2)
model.load_state_dict(torch.load('detect_position/baseline.pt'))
model.eval()

if not os.path.isdir('./db'):
    os.mkdir('./db')


#@profile
def get_points_webcam(capture):
    """
    camera takes a pic
    calls openpose + position detection
    records the time stamp, coordinates & position indicator in the csv
    """

    ret, img = capture.read()
    end = time.time()

    start = time.time()
    #print(img.shape)
    candidate, subset = body_estimation(img)
    end = time.time()
    print('openpose time is', end-start)

    lst_points, data = process_candidate(candidate)
    lst_points = sit_or_stand(lst_points, data)
    return lst_points, data

def process_candidate(candidate):
    candidate = dict_index(candidate) 
    # prepare the data to be used for the position detection model
    data = np.zeros((2, 18))
    for col in range(18):
        if col in candidate:
            x = candidate[col][0]
            y = candidate[col][1]
            data[0, col] = x
            data[1, col] = y
        else:
            data[0, col] = -1
            data[1, col] = -1
    
    # create a list of timestamp and 18 coordinates
    lst_points = [datetime.now()]
    for i in range(18):
        x = data[0][i]
        y = data[1][i]
       # point = (x, y)
        lst_points.append(x)
        lst_points.append(y)

    return lst_points, data

def sit_or_stand(lst_points, data):
    # if no points detected, y_pred = -1, meaning no person in view
    if all(p == -1 for p in lst_points[1:]):
        y_pred = -1
    else:
        # run the position detection model
        start = time.time()

        data = torch.Tensor(data)
        y_score = model(data.view(-1, 36)).detach().softmax(dim=1).numpy()   
        y_pred = np.argmax(y_score, axis=1).item()

        if y_pred == 0:
            print('stand')
        if y_pred == 1:
            print('sit')
        end = time.time()
        print('position classification time:', end - start)

    lst_points.append(y_pred)

    return lst_points







def record_points(db_path, lst_points):
    conn = sqlite3.connect(f'file:{db_path}?mode=rwc', uri=True)
    cur = conn.cursor()

    cols = '('
    for i in range(37):
        cols += '?, '
    cols += '?)'

    sql_statement = "INSERT INTO records VALUES " + cols
    cur.executemany(sql_statement, [tuple(lst_points)])
    conn.commit()
    conn.close()
    logger.info(f'Database created at {db_path}.')




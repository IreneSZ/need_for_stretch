import argparse
import copy
import os
import sqlite3
import time
import timeit
from datetime import datetime
from pathlib import Path
from sqlite3 import Error

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from op import model, util
from op.body import Body
from op.hand import Hand

from reader import Reader
from utils import dict_index
from model import Model


# @profile
def get_points_webcam(reader: Reader, model: Model):
    """
    camera takes a pic
    calls openpose + position detection
    records the time stamp, coordinates & position indicator in the csv
    """

    ret, img = reader.read()
    end = time.time()
    start = time.time()
    candidate, subset = model.estimate_body(img)
    end = time.time()
    logger.info(f'openpose time is {end-start}.')

    return process_candidate(candidate)


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
        lst_points.append(x)
        lst_points.append(y)

    return lst_points, data


def sit_or_stand(lst_points, data, model: Model):
    # if no points detected, y_pred = -1, meaning no person in view
    if all(p == -1 for p in lst_points[1:]):
        y_pred = -1
    else:
        # only run the position detection model when person is detected
        start = time.time()
        data = torch.Tensor(data)
        y_score = model.classify_sit_or_stand(
            data.view(-1, 36)).detach().softmax(dim=1).numpy()
        y_pred = np.argmax(y_score, axis=1).item()
        end = time.time()
        logger.info(f'position classification time is {end-start}.')

    pred2status = {
        0: 'stand',
        1: 'sit',
        -1: 'off screen'
    }
    end = time.time()
    logger.info(f'position classified as: {pred2status[y_pred]}.')

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
    cur.executemany(sql_statement, [lst_points])
    conn.commit()
    conn.close()
    logger.info(f'Database updated at {db_path}.')

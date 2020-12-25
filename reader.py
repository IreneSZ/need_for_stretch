from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2
from loguru import logger


class Reader(ABC):

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]: ...


class VideoReader(Reader):

    def __init__(self, video_capture):
        self._video_capture = video_capture
    
    def read(self):
        return self._video_capture.read()
    

class DebugReader(Reader):

    def __init__(self, img_folder: str):
        self._img_folder = img_folder
    
    def read(self):
        img_path = f'{self._img_folder}/squat1.jpg'
        logger.info(f'Debug: read image from {img_path}.')
        img = cv2.imread(f'{img_path}')
        img = cv2.resize(img, (320, 240)) 
        return True, img


from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import cv2
from loguru import logger
from pathlib import Path
from utils import get_img_name


class Reader(ABC):

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]: ...


class VideoReader(Reader):

    def __init__(self, debug: bool = False):
        self._debug = debug
        if debug:
            self._debug_folder = './debug/'
            folder = Path(self._debug_folder)
            folder.mkdir(parents=True, exist_ok=True)
    
    def read(self):
        capture = cv2.VideoCapture(-1)
        capture.set(3, 320)
        capture.set(4, 240)
        success, img = capture.read()
        if self._debug:
            img_path = f'{self._debug_folder}/{get_img_name()}'
            cv2.imwrite(img_path, img)
            logger.info(f'Debug mode, cam reader result saved to {img_path}.') 
        return success, img

class DebugReader(Reader):

    def __init__(self, img_folder: str):
        self._img_folder = img_folder
    
    def read(self):
        img_path = f'{self._img_folder}/squat1.jpg'
        logger.info(f'Debug: read image from {img_path}.')
        img = cv2.imread(f'{img_path}')
        img = cv2.resize(img, (320, 240)) 
        return True, img


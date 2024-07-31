import cv2
import numpy as np
import logging
from typing import NamedTuple

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)




class NanoTracker():
    def __init__(self, backbone, neckhead) -> None:
        params = cv2.TrackerNano_Params()
        params.backbone = backbone
        params.neckhead = neckhead
        self.tracker = cv2.TrackerNano_create(params)

        
    def calc(self, frame):
        midpoint = np.array([-1.0,-1.0])
        up_left = np.array([-1.0,-1.0])
        down_right = np.array([-1.0,-1.0])


        ok, newbox = self.tracker.update(frame)
        return ok, newbox

    def track(self, frame, bbox):
        try:
            self.tracker.init(frame, bbox)
        except Exception as e:
            print('Unable to initialize tracker with requested bounding box. Is there any object?')
            print(e)
            print('Try again ...')

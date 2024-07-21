import cv2
import numpy as np
import logging
from typing import NamedTuple

logging.basicConfig(level=logging.DEBUG)

log = logging.getLogger(__name__)

NANO_BACKBONE = "/workspaces/rome_ws/src/rome_application/submodules/visual_trackers/visual_trackers/nanotrack/nanotrack_backbone_sim.onnx"
NANO_HEADNECK = "/workspaces/rome_ws/src/rome_application/submodules/visual_trackers/visual_trackers/nanotrack/nanotrack_head_sim.onnx"


class NanoTracker():
    def __init__(self, width=640, height=360) -> None:
        params = cv2.TrackerNano_Params()
        params.backbone = NANO_BACKBONE
        params.neckhead = NANO_HEADNECK
        self.tracker = cv2.TrackerNano_create(params)

        
    def calc(self, frame):
        midpoint = np.array([-1.0,-1.0])
        up_left = np.array([-1.0,-1.0])
        down_right = np.array([-1.0,-1.0])


        ok, newbox = self.tracker.update(frame)
        return ok, newbox

    def track(self, frame, bbox):
        try:
            print("---")
            print(bbox)
            self.tracker.init(frame, bbox)
        except Exception as e:
            print('Unable to initialize tracker with requested bounding box. Is there any object?')
            print(e)
            print('Try again ...')

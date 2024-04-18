import cv2
import numpy as np

feature_params = dict(maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
 maxLevel = 2,
 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

WINDOW_NAME = "LK"
cv2.namedWindow(WINDOW_NAME)


class LKTracker():
    def __init__(self, width=640, height=360) -> None:
        self.old_gray = None
        self.mask = np.zeros((height, width))
        self.color = np.random.randint(0, 255, (100, 3))
        self.p0 = None
        self.tracking = False
        self.tracking_init = False
        self.lock_gate = (100, 100)
        self.track_x = None
        self.track_y = None

    def register_xxx(self, handler):
        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                handler(x,y)

        cv2.setMouseCallback(WINDOW_NAME, draw_rectangle)

    def track(self, x, y):
        self.tracking = not self.tracking

        if self.tracking:
            self.track_x = x
            self.track_y = y
            self.tracking_init = True

            

    def calc(self, frame):
        midpoint = (-1,-1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.tracking_init:
            self.tracking_init = False
            self.old_gray = frame_gray
            feature_mask = np.zeros_like(frame_gray)
            rect_width, rect_height = self.lock_gate
            top_left_pt = (self.track_x - rect_width // 2, self.track_y - rect_height // 2)
            bottom_right_pt = (self.track_x + rect_width // 2, self.track_y + rect_height // 2)
            feature_mask = cv2.rectangle(feature_mask, top_left_pt, bottom_right_pt, (255,255,255), -1)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=feature_mask, **feature_params)
            
            #TODO: remove
            self.mask = np.zeros_like(frame)

        if self.tracking:
            
            if self.old_gray is None:
                
                return
            
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                midpoint = np.mean(good_new, axis=0)
                good_old = self.p0[st==1]
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            frame = cv2.add(frame, self.mask)
            self.p0 = good_new.reshape(-1, 1, 2)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1) 
        self.old_gray = frame_gray.copy()
        return midpoint
        
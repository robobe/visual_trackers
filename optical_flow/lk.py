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

class LKTracker():
    def __init__(self) -> None:
        self.old_gray = None
        self.mask = None
        self.color = np.random.randint(0, 255, (100, 3))
        self.p0 = None

    def init_tracker():
        pass

    def calc(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None:
            self.old_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **feature_params)
            self.mask = np.zeros_like(frame)
            return
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, self.mask)
        self.get_logger().info("lk recv --")
        cv2.imshow('of', img)
        cv2.waitKey(1) 
        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
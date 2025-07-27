import cv2 as cv 
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, 
                smooth_landmarks=True, enable_segmentation=False, 
                smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.mode = static_image_mode
        self.mod_complex = model_complexity
        self.smooth_lm = smooth_landmarks
        self.enable_seg = enable_segmentation
        self.smooth_seg = smooth_segmentation
        self.detectConf = min_detection_confidence
        self.trackConf = min_tracking_confidence
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.mod_complex, self.smooth_lm, self.enable_seg,
                                    self.smooth_seg, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        self.results = self.pose.process(imgRGB) 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        land_mark_list = []
        if self.results.pose_landmarks:
            for id, land_mark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(land_mark.x * width), int(land_mark.y * height)
                land_mark_list.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 2, (0, 255, 0), cv.FILLED)
        return land_mark_list


def main():
    cap = cv.VideoCapture('Poses/Videos/5385817-hd_1080_1920_25fps.mp4')

    prevTime = 0
    currTime = 0
    
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        img = cv.resize(img, (480, 640))
        
        land_mark_list = detector.findPosition(img, draw=False)
        if len(land_mark_list) != 0:
            print(land_mark_list[11]) # isolate left shoulder
            cv.circle(img, (land_mark_list[11][1], land_mark_list[11][2]), 10, (255, 239, 0), cv.FILLED)
        
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime     
        cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)

        
        cv.imshow('Video', img)        
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    main()
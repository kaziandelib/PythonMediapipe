import cv2 as cv 
import mediapipe as mp
import time

class handDetect():
    def __init__(self, mode=False, maxHands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.track_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLM in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLM, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNumber=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for id, landmark in enumerate(hand.landmark):
                    height, width, channels = img.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([id, cx, cy])
                    if draw:
                        cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
        return landmark_list

def main():
    cap = cv.VideoCapture(0)
    prevTime = 0
    detector = handDetect()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)
        if len(landmark_list) != 0:
            print(landmark_list[12])
        
        currTime = time.time()
        fps = 1 / (currTime - prevTime) if (currTime - prevTime) != 0 else 0
        prevTime = currTime
        cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)
        
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('x'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

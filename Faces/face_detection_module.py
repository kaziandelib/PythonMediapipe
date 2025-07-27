import cv2 as cv 
import mediapipe as mp 
import time 

class faceDetector():
    def __init__(self, min_detection_confidence=0.5):
        self.detectConf = min_detection_confidence
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.detectConf)
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findFaces(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        self.results = self.face.process(imgRGB) 
        bounding_boxes = []
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bounding_boxClass = detection.location_data.relative_bounding_box
                image_height, image_width, image_channel = img.shape
                bounding_box = int(bounding_boxClass.xmin * image_width), int(bounding_boxClass.ymin * image_height), \
                            int(bounding_boxClass.width * image_width), int(bounding_boxClass.height * image_height)
                bounding_boxes.append([bounding_box, detection.score])
                if draw:
                    img = self.box(img, bounding_box)
                    cv.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], (bounding_box[1] - 20)), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), thickness=3)
            
        return img, bounding_boxes
        
    def box(self, img, bounding_box, l=40, t=10, rt=1):
        x, y, w, h = bounding_box
        x1, y1 = x + w, y + h
            
        cv.rectangle(img, bounding_box, (255, 255, 0), rt)
        # Top left
        cv.line(img, (x, y), (x + l, y), (255, 255, 0), t)
        cv.line(img, (x, y), (x, y + l), (255, 255, 0), t)

        # Top right
        cv.line(img, (x1, y), (x1 - l, y), (255, 255, 0), t)
        cv.line(img, (x1, y), (x1, y + l), (255, 255, 0), t)

        # Bottom left
        cv.line(img, (x, y1), (x + l, y1), (255, 255, 0), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 255, 0), t)
    
        # Bottom right
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 255, 0), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 255, 0), t)    
            
        return img
            
            
    
def main():
    cap = cv.VideoCapture('Faces/Videos/4153808-uhd_4096_2160_25fps.mp4')
    prevTime = 0
    detector = faceDetector()
    
    while True:
        success, img = cap.read()    
        img, bounding_boxes =  detector.findFaces(img)
        img = cv.resize(img, (800, 600))
        
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime     
        cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)
    
    
        cv.imshow('Video', img)        
        if cv.waitKey(1) & 0xFF == ord('x'):
            break
        
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
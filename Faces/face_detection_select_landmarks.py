import cv2 as cv 
import mediapipe as mp 
import time 

cap = cv.VideoCapture('Faces/Videos/4153808-uhd_4096_2160_25fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/4216631-uhd_3840_2160_30fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/6014532-uhd_4096_2160_24fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/8441652-uhd_4096_2160_25fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/7640077-hd_1920_1080_25fps.mp4')
# cap = cv.VideoCapture(0)

prevTime = 0
mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()    
    img = cv.resize(img, (400, 400))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    results = face.process(imgRGB) 
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bounding_boxClass = detection.location_data.relative_bounding_box
            image_height, image_width, image_channel = img.shape
            bounding_box = int(bounding_boxClass.xmin * image_width), int(bounding_boxClass.ymin * image_height), \
                           int(bounding_boxClass.width * image_width), int(bounding_boxClass.height * image_height)
            cv.rectangle(img, bounding_box, (255, 255, 0), thickness=3)
            cv.putText(img, f'{int(detection.score[0] * 100)}%', (bounding_box[0], (bounding_box[1] - 20)), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), thickness=3)
                
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime     
    cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)

    
    
    cv.imshow('Video', img)        
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()

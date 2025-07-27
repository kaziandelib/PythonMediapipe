import cv2 as cv 
import time
import pose_estimation_module as pem

# cap = cv.VideoCapture('Poses/Videos/4438080-hd_1920_1080_25fps.mp4')
cap = cv.VideoCapture(0)


prevTime = 0
currTime = 0
    
detector = pem.poseDetector()

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
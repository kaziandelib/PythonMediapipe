import cv2 as cv 
import time
import face_detection_module as fdm


cap = cv.VideoCapture(0)
prevTime = 0
detector = fdm.faceDetector()
    
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
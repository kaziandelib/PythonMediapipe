import cv2 as cv 
import time
import hand_tracking_module as htm

cap = cv.VideoCapture(0)
prevTime = 0
detector = htm.handDetect()
    
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    landmark_list = detector.findPosition(img)
    if len(landmark_list) != 0:
        print(landmark_list[12])
        cv.circle(img, (landmark_list[12][1], landmark_list[12][2]), 10, (255, 239, 0), cv.FILLED)
        
                
    currTime = time.time()
    fps = 1 / (currTime - prevTime) if (currTime - prevTime) != 0 else 0
    prevTime = currTime
    cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)



    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('x'):
        break
    
cap.release()
cv.destroyAllWindows()

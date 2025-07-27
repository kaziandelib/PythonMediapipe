import cv2 as cv 
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture('Poses/Videos/5385817-hd_1080_1920_25fps.mp4')

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    img = cv.resize(img, (480, 640))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    results = pose.process(imgRGB) 
    # print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime     
    cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)

    
    
    cv.imshow('Video', img)        
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()

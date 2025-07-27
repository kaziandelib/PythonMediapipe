import cv2 as cv 
import mediapipe as mp 
import time

cap = cv.VideoCapture('Faces/Videos/6014532-uhd_4096_2160_24fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/4153808-uhd_4096_2160_25fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/4216631-uhd_3840_2160_30fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/8441652-uhd_4096_2160_25fps.mp4')
# cap = cv.VideoCapture('Faces/Videos/7640077-hd_1920_1080_25fps.mp4')
# cap = cv.VideoCapture(0)

prevTime = 0
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=3)
mpDraw = mp.solutions.drawing_utils

draw_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()    
    img = cv.resize(img, (800, 600))
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
    results = face_mesh.process(imgRGB) 
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=draw_spec
            )
            for landmrk in face_landmarks.landmark:
                print(landmrk)


    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime     
    cv.putText(img, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), thickness=3)
    
    
    cv.imshow('Video', img)        
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv.destroyAllWindows()

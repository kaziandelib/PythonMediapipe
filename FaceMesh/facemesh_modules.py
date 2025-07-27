import cv2 as cv 
import mediapipe as mp 
import time

class FaceMeshDetect():
    def __init__(self, static_image_mode=False, max_num_faces=2, 
                refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        
        self.static = static_image_mode
        self.maxface = max_num_faces
        self.landmrk_refine = refine_landmarks
        self.detectCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh(self.static, self.maxface, self.landmrk_refine, self.detectCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.draw_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        
    
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
        self.results = self.face_mesh.process(self.imgRGB) 
        faces = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.draw_spec,
                        connection_drawing_spec=self.draw_spec
                    )
                face = []
                for id, landmrk in enumerate(face_landmarks.landmark):
                    # print(landmrk)
                    image_height, image_width, image_channel = img.shape
                    x, y = int(landmrk.x * image_width), int(landmrk.y * image_height)
                    print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture('Faces/Videos/6014532-uhd_4096_2160_24fps.mp4')
    prevTime = 0
    detector = FaceMeshDetect()
    
    while True:
        success, img = cap.read()    
        img, faces = detector.findFaceMesh(img, draw=True)
        img = cv.resize(img, (800, 600))
        if len(faces) != 0:
            print(faces)
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
import cv2
import numpy as np
import dlib

PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt2.xml')

'''Funcion que localiza la cara del ususario con OpenCV (haarcascade_frontalface_default, 
ademas dibuja un rectangulo en el area de la cara, si hay mas de una persona lo indica''' 
def detect_careto_OpenCV(imagen):
	texto="Mas de una cara detectada";
	img=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY);
	faces = face_classifier.detectMultiScale(img, 1.3, 5)
	if len(faces) > 1:
		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
	 	return imagen
	if faces is ():
		return imagen

	x,y,w,h=faces[0,:];
	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 2)
	return imagen

'''Funcion que localiza la cara del ususario con Dlib, ademas dibuja un 
rectangulo en el area de la cara, si hay mas de una persona lo indica''' 
def detect_careto_Dlib(imagen):
	texto="Mas de una cara detectada";
	faces=detector(imagen,1)
	if len(faces) > 1:
		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
		return imagen
	if len(faces) == 0:
		return imagen

	p=faces[0]
	x=p.left()
	y=p.top()
	w=p.right()-p.left()
	h=p.bottom()-p.top()
	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 2)
	return imagen

def obtener_puntos(imagen):
	faces=detector(imagen,1)
	if len(faces)!=1:
		return "error"
	return np.matrix([[i.x, i.y] for i in predictor(imagen,faces[0]).parts()]) #Transformamos de formato dlib.point a matriz para poder acceder a los datos mas facilmente

def dibujar_puntos(imagen, puntos):
    imagen = imagen.copy()
    for idx, point in enumerate(puntos):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(imagen, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(imagen, pos, 3, color=(0, 255, 255))
    return imagen


image_webcam=cv2.VideoCapture(0)

while True:
	ret,frame = image_webcam.read()
	puntos=obtener_puntos(frame)
	imprimir=dibujar_puntos(frame,puntos)
	
	cv2.imshow('Detector de Fatiga', imprimir)

	if cv2.waitKey(1)==13:
		break

# imagen=cv2.imread("images/pedrolo.jpg")
# puntos=obtener_puntos(imagen)
# imprimir=dibujar_puntos(imagen,puntos)
# cv2.imshow("Careto",imprimir)
# cv2.waitKey(0)

cv2.destroyAllWindows()
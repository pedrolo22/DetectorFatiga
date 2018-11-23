import cv2
import numpy as np
#import dlib

PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
#predictor = dlib.shape_predictor(PREDICTOR_PATH)
#detector = dlib.get_frontal_face_detector()
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt2.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

'''Funcion que localiza la cara del ususario con OpenCV (haarcascade_frontalface_default, 
ademas dibuja un rectangulo en el area de la cara, si hay mas de una persona lo indica''' 
def detect_careto_OpenCV(imagen):
	x,y,w,h = [0,0,0,0]
	print x
	texto="Mas de una cara detectada";
	img=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY);
	faces = face_classifier.detectMultiScale(img, 1.3, 5)
	if len(faces) > 1:
		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
	 	return imagen,x,y,w,h
	if faces is ():
		return imagen,x,y,w,h

	x,y,w,h=faces[0,:];
	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 2)
	return imagen,x,y,w,h

def detect_eyes_OpenCV(imagen):
	texto="Mas de una cara detectada";
	img=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY);
	eyes = eye_classifier.detectMultiScale(img, 1.3, 5)
	if len(eyes) < 2:
		return imagen
	print(eyes)
	ex1,ey1,ew1,eh1=eyes[0,:]
	ex2,ey2,ew2,eh2=eyes[1,:]
	cv2.rectangle(imagen,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),2)
	cv2.rectangle(imagen,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
	return imagen

'''Funcion que localiza la cara del ususario con Dlib, ademas dibuja un 
rectangulo en el area de la cara, si hay mas de una persona lo indica''' 
# def detect_careto_Dlib(imagen):
# 	texto="Mas de una cara detectada";
# 	faces=detector(imagen,1)
# 	if len(faces) > 1:
# 		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
# 		return imagen
# 	if len(faces) == 0:
# 		return imagen

# 	p=faces[0]
# 	x=p.left()
# 	y=p.top()
# 	w=p.right()-p.left()
# 	h=p.bottom()-p.top()
# 	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 2)
# 	return imagen

'''Funcion que devuelve matriz numpy con 68 puntos faciales'''
# def obtener_puntos(imagen):
# 	faces=detector(imagen,1)
# 	if len(faces)!=1:
# 		return "error"
# 	return np.matrix([[i.x, i.y] for i in predictor(imagen,faces[0]).parts()]) #Transformamos de formato dlib.point a matriz para poder acceder a los datos mas facilmente

# '''Funcion que recorre los 68 puntos faciales y los dibuja en pantalla con su respectivo indice'''
# def dibujar_puntos(imagen, puntos):
#     imagen = imagen.copy()
#     for index, punto in enumerate(puntos):
#     	if len(puntos)<68:
#     		break
#         pos = (punto[0, 0], punto[0, 1])
#         cv2.putText(imagen, str(index), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
#         cv2.circle(imagen, pos, 3, color=(0, 255, 255))
#     return imagen

# ''' Los ojos corresponden con los puntos del 36 al 47. Esta funcion obtiene una matriz con estos puntos y toma como argumento los 68 puntos faciales'''
# def obtener_puntos_ojos(puntos):
# 	puntos_ojos=np.zeros((12,2))
# 	aux=np.array([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])
# 	if len(puntos)<68:
# 		return puntos_ojos
# 	for index,i in enumerate(aux):
# 		puntos_ojos[index]=puntos[i]
# 	return puntos_ojos

# '''Esta funcion dibuja los puntos de los ojos (12 Puntos)'''
# def dibujar_puntos_ojos(imagen, puntos_ojos):
# 	imagen=imagen.copy()
# 	for index, punto in enumerate(puntos_ojos):
# 		if len(puntos)<12:
# 			break
# 		pos = (int(punto[0]), int(punto[1]))
# 		cv2.putText(imagen, str(index), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
# 		cv2.circle(imagen, pos, 3, color=(0, 255, 255))
# 	return imagen

# ''' La boca corresponde con los puntos del 48 al 67. Esta funcion obtiene una matriz con estos puntos y toma como argumento los 68 puntos faciales'''
# def obtener_puntos_boca(puntos):
# 	puntos_boca=np.zeros((21,2))
# 	aux=np.array([48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67])
# 	if len(puntos)<68:
# 		return puntos_boca
# 	for index,i in enumerate(aux):
# 		puntos_boca[index]=puntos[i]
# 	return puntos_boca

# '''Esta funcion dibuja los puntos de la boca (20 Puntos)'''
# def dibujar_puntos_boca(imagen, puntos_boca):
# 	imagen=imagen.copy()
# 	for index, punto in enumerate(puntos_boca):
# 		if len(puntos)<21:
# 			break
# 		pos = (int(punto[0]), int(punto[1]))
# 		cv2.putText(imagen, str(index), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
# 		cv2.circle(imagen, pos, 3, color=(0, 255, 255))
# 	return imagen



image_webcam=cv2.VideoCapture(0)

while True:
	ret,frame =image_webcam.read()
	frame_uMat=cv2.UMat(frame)
	#puntos=obtener_puntos(frame_uMat)
	im,x,y,w,h=detect_careto_OpenCV(frame_uMat)
	imprimir=detect_eyes_OpenCV(im)
	#crop=cv2.UMat(im,[x,y],[x+w,y+h])
	#imprimir=detect_eyes_OpenCV(crop)
	cv2.imshow('Detector de Fatiga', imprimir)

	if cv2.waitKey(1)==13:
		break

cv2.destroyAllWindows()
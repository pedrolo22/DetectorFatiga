#En este script vamos a crear un nuevo usuario a traves de la webCam

import cv2
import pruebaWebCam.py

image_webcam=cv2.VideoCapture(0)

while True:
	ret,frame = image_webcam.read()
	imagen=detect_careto(frame)
	cv2.imshow('Detector de Fatiga', imagen)

	if cv2.waitKey(1)==13:
		break


import cv2
import numpy as np
import time as t
import math
from matplotlib import pyplot as plt


kernel7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
kernel8=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
kernel9=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
kernel10=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
kernel11=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
kernel12=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
kernel13=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
kernel14=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(14,14))
kernel15=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
kernel16=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))
kernel17=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))

kernel31=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
kernel32=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(32,32))
kernel33=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33))
kernel34=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(34,34))
kernel35=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))
kernel36=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(36,36))
kernel37=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,37))
kernel38=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(38,38))
kernel39=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(39,39))
kernel40=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))




PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_alt2.xml')
face_classifier_LBP = cv2.CascadeClassifier('Haarcascades/lbpcascade_frontalface.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
cv2.ocl.setUseOpenCL(True)

'''Funcion que localiza la cara del ususario con OpenCV (haarcascade_frontalface_default, 
ademas dibuja un rectangulo en el area de la cara, si hay mas de una persona lo indica''' 
def detect_careto_OpenCV(imagen):
	x,y,w,h = [0,0,0,0]
	detect=False;
	texto="Mas de una cara detectada";
	#img=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY);
	img=imagen
	faces = face_classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
	if len(faces) > 1:
		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
		detect=False
		return imagen,x,y,w,h,detect

	if faces is ():
		return imagen,x,y,w,h,detect

	detect=True
	x,y,w,h=faces[0,:]
	#cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 3)
	return imagen,x,y,w,h,detect

def detect_eyes_OpenCV(imagen):
	
	#imagen=cv2.equalizeHist(imagen)
	eyes = eye_classifier.detectMultiScale(imagen, scaleFactor=1.1, minNeighbors=3)
	detect_eyes=True
	eye1=0
	eye2=0
	if len(eyes) < 2:
		detect_eyes=False
		return imagen,eye1,eye2,detect_eyes
	eye1=eyes[0,:]
	eye2=eyes[1,:]
	ex1,ey1,ew1,eh1=eye1
	ex2,ey2,ew2,eh2=eye2
	#cv2.rectangle(imagen,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),1)
	#cv2.rectangle(imagen,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),1)
	return imagen,eye1,eye2,detect_eyes


def proy_bin(imagen):
	img_suav=cv2.GaussianBlur(imagen,(3,3),0)
	img_equal=cv2.equalizeHist(img_suav)
	proy_ver=np.sum(255-img_equal,0)
	proy_hor=np.sum(255-img_equal,1)
	index_ver=sum(proy_ver>3500)
	index_hor=sum(proy_hor>3500)
	apertura=float(index_hor)/float(index_ver)
	print(index_ver)
	print(index_hor)
	print(apertura)

	plt.subplot(131)
	plt.plot(proy_hor)
	plt.title('proy_hor')
	plt.subplot(132)
	plt.plot(proy_ver)
	plt.title('proy_ver')
	plt.subplot(133)
	plt.imshow(img_equal,cmap='gray')
	plt.title('ojo')
	plt.show()
	return apertura

def morf_proc(imagen):

	if(imagen.shape[0]==0 or imagen.shape[1]==0):
		return 0
	else:

		img_rgb=cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR);
		img_suav=cv2.GaussianBlur(imagen,(3,3),0)
		img1=cv2.morphologyEx(img_suav, cv2.MORPH_CLOSE, kernel9)
		img2=cv2.morphologyEx(img1, cv2.MORPH_CLOSE,kernel15)
		img3=cv2.subtract(img2,img1)
		umbral1=np.amax(img3)*0.75
		ret,img_bw=cv2.threshold(img3, umbral1,255, cv2.THRESH_BINARY)
		img4=cv2.dilate(img_bw,np.ones(5))
		img5=cv2.subtract(img4,img_bw) #imagen con el contorno umbralizado

		#Proceso para aproximar a una elipse
		# contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(img_bw, contours, -1, (0,255,0), 1)
		# cnt=contours[0]
		# ellipse=cv2.fitEllipse(cnt)	
		# print(ellipse)

		#Proceso para encontrar el centro de masa del contorno de la pupila
		im_contorno=np.where(img5==255)
		coord_y=np.sum(im_contorno[0])/(im_contorno[0].shape[0])
		coord_x=np.sum(im_contorno[1])/(im_contorno[1].shape[0])
		distancia=np.where(im_contorno[1]==coord_x)
		xcol=im_contorno[0]
		aux=[]
		for i in distancia[0]:
		    aux.append(xcol[i])
		if(len(aux) != 0):
			apertura=max(aux)-min(aux) #Valor de apertura vertical del ojo en funcion de la pupila
		else:
			apertura=0
		print('Apertura Ojo',apertura)
		cv2.drawMarker(img5, (int(coord_x), int(coord_y)), (255,255,255), cv2.MARKER_CROSS,  markerSize = 2)

		cv2.imshow('Apertura1',img1)
		cv2.imshow('Apertura2',img2)
		cv2.imshow('1-2',img3)
		cv2.imshow('Umbralizacion',img_bw)
		cv2.imwrite('./capturas/elipse.png',img_rgb)
		cv2.imwrite('./capturas/contorno_pupila.png',img5)
		cv2.imwrite('./capturas/ojo_suav.jpg',img_suav)
		cv2.imshow('Dilatacion',img4)
		cv2.imshow('Contorno Pupila', img5)

	return apertura

def morf_proc_mouth(imagen):


	img_suav=cv2.GaussianBlur(imagen,(3,3),0)
	img1=cv2.morphologyEx(img_suav, cv2.MORPH_CLOSE, kernel32)
	img2=cv2.morphologyEx(img1, cv2.MORPH_CLOSE,kernel38)
	img3=cv2.subtract(img2,img1)
	umbral1=np.amax(img3)*0.75
	ret,img_bw=cv2.threshold(img3, umbral1,255, cv2.THRESH_BINARY)
	img4=cv2.dilate(img_bw,np.ones(5))
	img5=cv2.subtract(img4,img_bw) #imagen con el contorno umbralizado
	im_contorno=np.where(img5==255)
	coord_y=np.sum(im_contorno[0])/(im_contorno[0].shape[0])
	coord_x=np.sum(im_contorno[1])/(im_contorno[1].shape[0])
	distancia=np.where(im_contorno[1]==coord_x)
	xcol=im_contorno[0]
	aux=[]
	for i in distancia[0]:
	    aux.append(xcol[i])
	if(len(aux) != 0):
		apertura=max(aux)-min(aux) #Valor de apertura vertical de la boca
	else:
		apertura=0
	print('Apertura Boca',apertura)
	cv2.drawMarker(img5, (int(coord_x), int(coord_y)), (255,255,255), cv2.MARKER_CROSS,  markerSize = 2)

	# cv2.imshow('Apertura1',img1)
	# cv2.imshow('Apertura2',img2)
	# cv2.imshow('1-2',img3)
	# cv2.imshow('Umbralizacion',img_bw)
	# cv2.imshow('Dilatacion',img4)
	# cv2.imshow('Contorno Boca', img5)

	return apertura
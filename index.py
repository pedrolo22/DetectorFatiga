import cv2
import numpy as np
import time as t
from matplotlib import pyplot as plt
#import dlib

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
	faces = face_classifier.detectMultiScale(img, 1.3, 5)
	if len(faces) > 1:
		cv2.putText(imagen, texto, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)
		detect=False
		return imagen,x,y,w,h,detect

	if faces is ():
		return imagen,x,y,w,h,detect

	detect=True
	x,y,w,h=faces[0,:]
	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 1)
	return imagen,x,y,w,h,detect

def detect_eyes_OpenCV(imagen):
	#img=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY);
	eyes = eye_classifier.detectMultiScale(imagen, 1.3, 5)
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

def ellipse_aprox(imagen):
	img_suav=cv2.GaussianBlur(imagen,(3,3),0)
	img_equal=cv2.equalizeHist(img_suav)
	img_bw=cv2.adaptiveThreshold(img_equal, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,2)
	contours,hierarchy=cv2.findContours(img_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(img_bw,contours,-1,(0,255,0),3)
	#cv2.ellipse(imagen,(0,255,0),2)
	cv2.imshow('Ojo binarizado',img_equal)

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

start_time=t.time()
im=cv2.imread('images/ojos_entre_HD.jpg',0)
im_resize=cv2.UMat(cv2.resize(im,(320,180)))

face_UMat,fx,fy,fw,fh,fdetect=detect_careto_OpenCV(im_resize)
face=cv2.UMat.get(face_UMat)
cv2.imshow('pedrolo',im_resize)
cv2.imwrite('./capturas/im_resize.jpg',face)
face_time=t.time()-start_time
print('Tiempo ejecucion cara',face_time)

ROI_face=face[fy:fy+fh , fx:fx+fw]
cv2.imshow('face',ROI_face)
cv2.imwrite('./capturas/face.jpg',ROI_face)
dimY,dimX=ROI_face.shape
print('Tamano cara',dimY,dimX)
ROI_eyes=ROI_face[int(dimX*0.25):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
ROI_mouth=ROI_face[int(dimX*0.7):int(dimX*0.95),int(dimY*0.2):int(dimY*0.8)]
cv2.imshow('mouth', ROI_mouth)
cv2.imwrite('./capturas/mouth.jpg',ROI_mouth)
print('Tamano boca',ROI_mouth.shape)
print('Tiempo ejecucion crops',t.time()-start_time)

eyes,eye1,eye2,detect_eyes=detect_eyes_OpenCV(ROI_eyes)
ex1,ey1,ew1,eh1=eye1
ex2,ey2,ew2,eh2=eye2
ROI_eye1=eyes[ey1:ey1+eh1 ,ex1:ex1+ew1]
ROI_eye2=eyes[ey2:ey2+eh2 ,ex2:ex2+ew2]
am=proy_bin(ROI_eye1)
cv2.imshow('eyes',eyes)
cv2.imwrite('./capturas/eyes.jpg',ROI_eyes)
cv2.imwrite('./capturas/eye1.jpg',ROI_eye1)
cv2.imwrite('./capturas/eye2.jpg',ROI_eye2)
print('Tamano ojo',ROI_eye1.shape)
print('Tiempo ejecucion ojos',t.time()-start_time)

cv2.waitKey(0)

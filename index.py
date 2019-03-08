import cv2
import numpy as np
import time as t
import math
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

def morf_proc(imagen):

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

	img_rgb=cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB);
	img_suav=cv2.GaussianBlur(imagen,(3,3),0)
	img1=cv2.morphologyEx(img_suav, cv2.MORPH_CLOSE, kernel9)
	img2=cv2.morphologyEx(img1, cv2.MORPH_CLOSE,kernel15)
	img3=cv2.subtract(img2,img1)
	umbral1=np.amax(img3)*0.75
	ret,img_bw=cv2.threshold(img3, umbral1,255, cv2.THRESH_BINARY)
	img4=cv2.dilate(img_bw,np.ones(3))
	img5=cv2.subtract(img4,img_bw)
	contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img_bw, contours, -1, (0,255,0), 1)
	cnt=contours[0]
	ellipse=cv2.fitEllipse(cnt)	
	print(ellipse)

	cv2.imshow('Apertura1',img1)
	cv2.imshow('Apertura2',img2)
	cv2.imshow('1-2',img3)
	cv2.imshow('Umbralizacion',img_bw)
	cv2.imwrite('./capturas/elipse.png',img_rgb)
	cv2.imshow('Dilatacion',img4)
	cv2.imshow('Contorno Pupila', imagen)

	return ellipse



start_time=t.time()
im_rgb=cv2.imread('images/ojos_abiertos_HD.jpg')
im_rgb_resize=cv2.resize(im_rgb,(640,360))
im=cv2.cvtColor(im_rgb,cv2.COLOR_RGB2GRAY)
im_resize=cv2.UMat(cv2.resize(im,(640,360)))

face_UMat,fx,fy,fw,fh,fdetect=detect_careto_OpenCV(im_resize)
face=cv2.UMat.get(face_UMat)
#cv2.imshow('pedrolo',im_resize)
face_time=t.time()-start_time
print('Tiempo ejecucion cara',face_time)

ROI_face=face[fy:fy+fh , fx:fx+fw]
#cv2.imshow('face',ROI_face)
dimY,dimX=ROI_face.shape
print('Tamano cara',dimY,dimX)
ROI_eyes=ROI_face[int(dimX*0.25):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
ROI_mouth=ROI_face[int(dimX*0.7):int(dimX*0.95),int(dimY*0.2):int(dimY*0.8)]
#cv2.imshow('mouth', ROI_mouth)

print('Tamano boca',ROI_mouth.shape)
print('Tiempo ejecucion crops',t.time()-start_time)

eyes,eye1,eye2,detect_eyes=detect_eyes_OpenCV(ROI_eyes)
ex1,ey1,ew1,eh1=eye1
ex2,ey2,ew2,eh2=eye2
expand_eyes=5
ROI_eye1=eyes[(ey1-expand_eyes):(ey1+eh1+expand_eyes) ,(ex1-expand_eyes):(ex1+ew1+expand_eyes)]
ROI_eye2=eyes[(ey2-expand_eyes):(ey2+eh2+expand_eyes) ,(ex2-expand_eyes):(ex2+ew2+expand_eyes)]

# Procesamiento morfologico y aproximacion de elipse a pupila
ellipse=morf_proc(ROI_eye1)
center,axis,angle=ellipse
a=float(axis[1])
b=float(axis[0])
excentricidad=math.sqrt(a**2-b**2)/a
center=(center[0]+ex1+fx+dimY*0.1-expand_eyes,center[1]+ey1+fy+dimX*0.25-expand_eyes)
ellipse=[center,axis,angle]
cv2.ellipse(im_rgb_resize,tuple(ellipse),(0,255,0),1)

print('Semieje menor',axis[0])
print('Semieje mayor',axis[1])
print('Excentricidad', excentricidad)
cv2.putText(im_rgb_resize, 'Excentricidad= '+ str(excentricidad), (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)


cv2.imwrite('./capturas/im_resize.jpg',face)
cv2.imwrite('./capturas/face.jpg',ROI_face)
cv2.imwrite('./capturas/mouth.jpg',ROI_mouth)
cv2.imwrite('./capturas/eyes.jpg',ROI_eyes)
cv2.imwrite('./capturas/eye1.jpg',ROI_eye1)
cv2.imwrite('./capturas/eye2.jpg',ROI_eye2)
cv2.imwrite('./capturas/elipse_rgb.png',im_rgb_resize)

print('Tamano ojo',ROI_eye1.shape)
print('Tiempo ejecucion ojos',t.time()-start_time)

cv2.waitKey(0)

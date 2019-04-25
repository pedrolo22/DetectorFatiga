import cv2
import numpy as np
import time as t
import math
from matplotlib import pyplot as plt
import functions as fun
#import dlib

start_time=t.time()
im_rgb=cv2.imread('images/bostezo3.png')
im=cv2.cvtColor(im_rgb,cv2.COLOR_RGB2GRAY)

y,x=im.shape
if ((float(x)/float(y))==(float(4)/float(3))):
	im_rgb_resize=cv2.resize(im_rgb,(600,450))
	im_resize=cv2.UMat(cv2.resize(im,(600,450)))
if((float(x)/float(y))==(float(16)/float(9))):
	im_rgb_resize=cv2.resize(im_rgb,(640,360))
	im_resize=cv2.UMat(cv2.resize(im,(640,360)))
else:
	im_rgb_resize=im_rgb
	im_resize=cv2.UMat(im)


face_UMat,fx,fy,fw,fh,fdetect=fun.detect_careto_OpenCV(im_resize)
face=cv2.UMat.get(face_UMat)

im_resize_gray_rgb=cv2.cvtColor(im_resize,cv2.COLOR_GRAY2RGB)
cv2.rectangle(im_resize_gray_rgb,(fx,fy) , (fx+fw,fy+fh), (127,0,255), 2)
#cv2.imshow('pedrolo',im_resize)
face_time=t.time()-start_time
print('Tiempo ejecucion cara',face_time)

ROI_face=face[fy:fy+fh , fx:fx+fw]
#cv2.imshow('face',ROI_face)
dimY,dimX=ROI_face.shape
print('Tamano cara',dimY,dimX)

ROI_eyes=ROI_face[int(dimX*0.15):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
ROI_mouth=ROI_face[int(dimX*0.6):int(dimX*1),int(dimY*0.2):int(dimY*0.8)]

# ROI_eyes=ROI_face[int(dimX*0.25):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
# ROI_mouth=ROI_face[int(dimX*0.7):int(dimX*1),int(dimY*0.2):int(dimY*0.8)]

cv2.imshow('Cara',ROI_face)
cv2.imshow('Ojos',ROI_eyes)
cv2.imshow('Boca',ROI_mouth)


print('Tamano boca',ROI_mouth.shape)
print('Tiempo ejecucion crops',t.time()-start_time)

#Procesamiento para conocer el estado de los ojos

eyes,eye1,eye2,detect_eyes=fun.detect_eyes_OpenCV(ROI_eyes)
if detect_eyes==1:
	
	ex1,ey1,ew1,eh1=eye1
	ex2,ey2,ew2,eh2=eye2
	expand_eyes=5
	ROI_eye1=eyes[(ey1-expand_eyes):(ey1+eh1+expand_eyes) ,(ex1-expand_eyes):(ex1+ew1+expand_eyes)]
	ROI_eye2=eyes[(ey2-expand_eyes):(ey2+eh2+expand_eyes) ,(ex2-expand_eyes):(ex2+ew2+expand_eyes)]

	img_rgb_eyes=cv2.cvtColor(eyes, cv2.COLOR_GRAY2RGB)
	cv2.rectangle(img_rgb_eyes,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),1)
	cv2.rectangle(img_rgb_eyes,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),1)

	#Proyeccion binaria

	apertura=fun.proy_bin(ROI_eye2)
	
	# Procesamiento morfologico y aproximacion de elipse a pupila

	# ellipse=fun.morf_proc(ROI_eye2)
	# cv2.ellipse(ROI_eye2,tuple(ellipse),(255,255,255),1)
	# center,axis,angle=ellipse
	# a=float(axis[1])
	# b=float(axis[0])
	# excentricidad=math.sqrt(a**2-b**2)/a
	# center=(center[0]+ex2+fx+dimY*0.1-expand_eyes,center[1]+ey2+fy+dimX*0.15-expand_eyes)
	# ellipse=[center,axis,angle]
	# cv2.ellipse(im_rgb_resize,tuple(ellipse),(0,255,0),1)


	# print('Semieje menor',axis[0])
	# print('Semieje mayor',axis[1])
	# print('Excentricidad', excentricidad)
	# cv2.putText(im_rgb_resize, 'Excentricidad= '+ str(excentricidad), (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)

	cv2.imwrite('./capturas/im_orig.jpg', im_rgb)
	cv2.imwrite('./capturas/im_byw.jpg', im)
	cv2.imwrite('./capturas/im_byw_face.jpg', im_resize_gray_rgb)
	cv2.imwrite('./capturas/im_resize.jpg',face)
	cv2.imwrite('./capturas/face.jpg',ROI_face)
	cv2.imwrite('./capturas/mouth.jpg',ROI_mouth)
	cv2.imwrite('./capturas/eyes.jpg',ROI_eyes)
	cv2.imwrite('./capturas/eyes_eqal.jpg',img_rgb_eyes)
	cv2.imwrite('./capturas/eye1.jpg',ROI_eye1)
	cv2.imwrite('./capturas/eye2.jpg',ROI_eye2)
	cv2.imwrite('./capturas/elipse_rgb.png',im_rgb_resize)

	print('Tamano ojo',ROI_eye1.shape)
	print('Tiempo ejecucion ojos',t.time()-start_time)

else:
	print('No se ha detectado ningun ojo')


#Procesamiento para conocer el estado de la boca
cv2.imshow('mouth', ROI_mouth)
print(ROI_mouth.shape)
im_mouth,mouth,detect_mouth=fun.detect_mouth_OpenCV(im)
cv2.imshow('mouth rectangle', im_mouth)
apertura_boca=fun.morf_proc_mouth(ROI_mouth)





cv2.waitKey(0)

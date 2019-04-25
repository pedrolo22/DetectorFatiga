import cv2
import numpy as np
import time as t
import math
from matplotlib import pyplot as plt
import functions as fun
#import dlib

start_time=t.time()
im_rgb=cv2.imread('images/ojos_entre_HD.jpg')
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


#Deteccion de cara

face_UMat,fx,fy,fw,fh,fdetect=fun.detect_careto_OpenCV(im_resize)

if(fdetect==False):
	print('No se detecta rostro')
else:
	face=cv2.UMat.get(face_UMat)
	face_time=t.time()-start_time
	print('Tiempo ejecucion cara',face_time)
	ROI_face=face[fy:fy+fh , fx:fx+fw]
	dimY,dimX=ROI_face.shape
	ROI_eyes=ROI_face[int(dimX*0.15):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
	ROI_mouth=ROI_face[int(dimX*0.6):int(dimX*1),int(dimY*0.2):int(dimY*0.8)]

	#Deteccion de ojos
	eyes,eye1,eye2,detect_eyes=fun.detect_eyes_OpenCV(ROI_eyes)

	if(detect_eyes==False):
		apertura=0
		print('No se detecta ojo')
	else:
		ex1,ey1,ew1,eh1=eye1
		ex2,ey2,ew2,eh2=eye2
		expand_eyes=5
		ROI_eye1=eyes[(ey1-expand_eyes):(ey1+eh1+expand_eyes) ,(ex1-expand_eyes):(ex1+ew1+expand_eyes)]
		ROI_eye2=eyes[(ey2-expand_eyes):(ey2+eh2+expand_eyes) ,(ex2-expand_eyes):(ex2+ew2+expand_eyes)]
		eyes_time=t.time()-start_time
		print('Tiempo ejecucion ojos',eyes_time)
		#Apertura de ojos
		apertura=fun.morf_proc(ROI_eye2)
		apertura_time=t.time()-start_time
		print('Tiempo ejecucion proy_bin',apertura_time)

	#Apertura de boca
	apertura_boca=fun.morf_proc_mouth(ROI_mouth)
	apertura_time=t.time()-start_time
	print('Tiempo ejecucion apertura boca',apertura_time)

	print('Apertura ojos: ',apertura)
	print('Apertura boca: ',apertura_boca)
	cv2.waitKey(0)
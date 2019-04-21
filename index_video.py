import cv2
import numpy as np
import time as t
import math
from matplotlib import pyplot as plt
import functions as fun
#import dlib

start_time=t.time()
cap = cv2.VideoCapture('dataset/T001/T001-001.avi1.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_resize_noUmat=cv2.resize(im_gray,(360,360))
    cv2.imwrite('dataset/T001/captura.png',im_resize_noUmat)
    im_resize=cv2.UMat(cv2.resize(im_gray,(360,360)))

    face_UMat,fx,fy,fw,fh,fdetect=fun.detect_careto_OpenCV(im_resize)
    if (fdetect==1):
    	cv2.rectangle(im_resize_noUmat,(fx,fy) , (fx+fw,fy+fh), (127,0,255), 1)
    	face=cv2.UMat.get(face_UMat)
    	ROI_face=face[fy:fy+fh , fx:fx+fw]
    	dimY,dimX=ROI_face.shape
    	ROI_eyes=ROI_face[int(dimX*0.25):int(dimX*0.55),int(dimY*0.1):int(dimY*0.9)]
    	ROI_mouth=ROI_face[int(dimX*0.7):int(dimX*1),int(dimY*0.2):int(dimY*0.8)]
    	eyes,eye1,eye2,detect_eyes=fun.detect_eyes_OpenCV(ROI_eyes)

    	if detect_eyes==1:
    		ex1,ey1,ew1,eh1=eye1
    		ex2,ey2,ew2,eh2=eye2
    		expand_eyes=5
    		ROI_eye1=eyes[(ey1-expand_eyes):(ey1+eh1+expand_eyes) ,(ex1-expand_eyes):(ex1+ew1+expand_eyes)]
    		ROI_eye2=eyes[(ey2-expand_eyes):(ey2+eh2+expand_eyes) ,(ex2-expand_eyes):(ex2+ew2+expand_eyes)]
    		cv2.rectangle(im_resize_noUmat,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),1)
    		cv2.rectangle(im_resize_noUmat,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),1)

    		# Procesamiento morfologico y aproximacion de elipse a pupila
    		apertura_ojo=fun.morf_proc(ROI_eye1)
    		# center,axis,angle=ellipse
    		# a=float(axis[1])
    		# b=float(axis[0])
    		# excentricidad=math.sqrt(a**2-b**2)/a
    		# center=(center[0]+ex1+fx+dimY*0.1-expand_eyes,center[1]+ey1+fy+dimX*0.25-expand_eyes)
    		# ellipse=[center,axis,angle]
    		# cv2.ellipse(im_rgb_resize,tuple(ellipse),(0,255,0),1)

    	else:
    		cv2.putText(frame, 'No se detecta ningun ojo', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)

    	#Procesamiento para conocer el estado de la boca
    	apertura_boca=fun.morf_proc_mouth(ROI_mouth)
    else:
    	cv2.putText(frame, 'No se detecta ninguna cara', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (127,0,255), 2)

	
    cv2.imshow('Video',im_resize_noUmat)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













# cv2.waitKey(0)
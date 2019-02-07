import cv2
import numpy as np
import time
from matplotlib import pyplot as plt
#import dlib

PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
#predictor = dlib.shape_predictor(PREDICTOR_PATH)
#detector = dlib.get_frontal_face_detector()
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
	cv2.rectangle(imagen,(x,y) , (x+w,y+h), (127,0,255), 2)
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
	cv2.rectangle(imagen,(ex1,ey1),(ex1+ew1,ey1+eh1),(0,255,0),1)
	cv2.rectangle(imagen,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),1)
	return imagen,eye1,eye2,detect_eyes

def elim_ruido(imagen):
	dim=3
	filtro=np.ones((dim,dim),np.float32)/(dim*dim)
	im_suav=cv2.filter2D(imagen,-1,filtro)
	return im_suav


def procesado_ojo(ojo):
	ojo_bw=cv2.cvtColor(ojo,cv2.COLOR_BGR2GRAY)
	ojo_suav=elim_ruido(ojo_bw)
	#ojo_equal=cv2.equalizeHist(ojo_suav)
	umbral,ojo_umbr=cv2.threshold(ojo_suav,70,255,cv2.THRESH_BINARY_INV)
	kernel = np.ones((5,5),np.uint8)
	ojo_erod=cv2.erode(ojo_umbr, kernel, 5)
	proye_hor=np.sum(ojo_erod,0)
	proye_ver=np.sum(ojo_erod,1)
	umbral_hor=0.8*max(proye_hor)
	umbral_ver=0.3*max(proye_ver)
	index_hor=(proye_hor>umbral_hor)
	result=index_hor*proye_ver
	index_ver=(proye_ver>umbral_ver)
	pixeles_hor=np.float(sum(index_hor))
	pixeles_ver=np.float(sum(index_ver))
	apertura=pixeles_ver/pixeles_hor

	plt.subplot(241)
	plt.imshow(ojo,cmap='gray')
	plt.title('Ojo original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(242)
	plt.imshow(ojo_suav,cmap='gray')
	plt.title('Ojo tras suavizar')
	plt.subplot(243)
	plt.imshow(ojo_suav,cmap='gray')
	plt.title('Ojo tras ecualizar')
	plt.subplot(244)
	plt.imshow(ojo_umbr,cmap='gray')
	plt.title('Ojo tras umbralizar')
	plt.subplot(245)
	plt.imshow(ojo_erod,cmap='gray')
	plt.title('Ojo tras erosion')
	plt.subplot(246)
	plt.plot(proye_hor)
	plt.title('Proyeccion horizontal')
	plt.subplot(247)
	plt.plot(proye_ver)
	plt.title('Proyeccion vertical')
	plt.subplot(248)
	plt.plot(result)
	plt.title('Result')
	plt.show()

	return apertura


#Imprimir Cara y ojos
image_webcam=cv2.VideoCapture(0)
apertura=[]
while True:
	t=time.time()
	ret,frame =image_webcam.read()
	frame_uMat=cv2.UMat(frame)
	im,x,y,w,h,detect=detect_careto_OpenCV(frame_uMat)
	if detect==False:
		imp=im
	else:
		crop=cv2.UMat.get(im)
		crop=crop[y:y+h,x:x+w]
		imp,eye_1,eye_2,detect_eyes=detect_eyes_OpenCV(crop)
		if detect_eyes==False:
			imp=im
		else:
			ex1,ey1,ew1,eh1=eye_1
			ex2,ey2,ew2,eh2=eye_2
			#ojo1=imp[ey1:ey1+ew1,ex1:ex1+eh1]
			ojo2=imp[ey2:ey2+ew2,ex2:ex2+eh2]
			apertura_temp=procesado_ojo(ojo2)
			print(apertura_temp)
			apertura.append(apertura_temp)
			
			
			

	
	cv2.imshow('Detector de Fatiga',imp)

	if cv2.waitKey(1)==13:
		break

cv2.destroyAllWindows()
#!/bin/python
# mengimpor pustaka
import numpy as np
import cv2
import math

url = "http://server.swadexi.com/up/iot2.php"

# memproses gesture
hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

# rekaman video
cap = cv2.VideoCapture(0)

while 1:
	ret, img = cap.read()
	blur = cv2.GaussianBlur(img,(5,5),0) # bluring gambar 
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) # RGB(red,green blue) //merah, hilau ,biru
	retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # smader window image
	hand = hand_cascade.detectMultiScale(thresh1, 1.3, 5) # mendeteksi gambar yang dihasilkan di output smader
	mask = np.zeros(thresh1.shape, dtype = "uint8") # membuat lapisan 
	for (x,y,w,h) in hand: # 
		cv2.rectangle(img,(x,y),(x+w,y+h), (122,122,0), 2) 
		cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)
	img2 = cv2.bitwise_and(thresh1, mask)
	final = cv2.GaussianBlur(img2,(7,7),0)	
	contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(img, contours, 0, (255,255,0), 3)
	cv2.drawContours(final, contours, 0, (255,255,0), 3)

	if len(contours) > 0:
		cnt=contours[0]
		hull = cv2.convexHull(cnt, returnPoints=False)
		# mencari defeksional dan menghitung defeksi
		defects = cv2.convexityDefects(cnt, hull)
		count_defects = 0
		# tangan yang digunakan untuk mendapatkan kode
		# dengan sudut > 90 derajat 
		if defects!= None:
			for i in range(defects.shape[0]):
				p,q,r,s = defects[i,0]
				finger1 = tuple(cnt[p][0])
				finger2 = tuple(cnt[q][0])
				dip = tuple(cnt[r][0])
				# panjang semua yang berada pada segitiga
				a = math.sqrt((finger2[0] - finger1[0])**2 + (finger2[1] - finger1[1])**2)
				b = math.sqrt((dip[0] - finger1[0])**2 + (dip[1] - finger1[1])**2)
				c = math.sqrt((finger2[0] - dip[0])**2 + (finger2[1] - dip[1])**2)
				# apply cosine rule here
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29
				# sudut elevasi > 90 dan kecerahan rendah 
				if angle <= 90:
				    count_defects += 1
		# definisi gerakan tangan 
		if count_defects == 1:
			cv2.putText(img,"kode 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
			url = "http://server.swadexi.com/up/iot2.php?ruang=102&kode=1"
		elif count_defects == 2:
			url = "http://server.swadexi.com/up/iot2.php?ruang=102&kode=2"
			cv2.putText(img,"kode 3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 3:
			url = "http://server.swadexi.com/up/iot2.php?ruang=102&kode=3"
			cv2.putText(img,"kode 4", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
		elif count_defects == 4:
			url = "http://server.swadexi.com/up/iot2.php?ruang=102&kode=4"
			cv2.putText(img,"kode 5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
	cv2.imshow('SMADER',img)
	

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()
             
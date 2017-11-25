import numpy as np
import cv2
from utils import *

def get_sift(img, kp = [], mode = "gray", mask = None):
	img = denormalize_image(img)
	img = np.array(img*255, dtype="uint8")
	if mode == "gray":
		gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp,des = sift.compute(gray,kp)
		return des.flatten()
		# img=cv2.drawKeypoints(gray,kp, None)
		# cv2.imshow('result', img), cv2.waitKey(0)
		# cv2.destroyWindow("result")
	elif mode == "color":
		# image chnnel mask
		hist1 = cv2.calcHist([img],[0],mask,[256],[0,256]).T[0]
		hist2 = cv2.calcHist([img],[1],mask,[256],[0,256]).T[0]
		hist3 = cv2.calcHist([img],[2],mask,[256],[0,256]).T[0]
		# plt.hist(gray_img.ravel(),256,[0,256])
		# plt.title('Histogram for gray scale picture')
		# plt.show()
		return np.append(np.append(hist1, hist2), hist3)

def generate_kp(features_line):
	kp = []
	for i in range(int(len(features_line)/2)):
		left_eye = np.array([features_line[0], features_line[1]])
		right_eye = np.array([features_line[2], features_line[3]])
		nose = np.array([features_line[4], features_line[5]])
		face_line = left_eye - right_eye
		kp_scale = np.linalg.norm(face_line)/2
		kp_angle = np.degrees(np.arctan2(face_line[1], face_line[0]))
		for i in range(8):
			tmp_x = features_line[2*i]
			tmp_y = features_line[2*i+1]
			kp.append(cv2.KeyPoint(tmp_x, tmp_y, _size=kp_scale, _angle=kp_angle))

		# mask
		center = (left_eye + right_eye + nose)/3
		face_scale = np.linalg.norm(face_line)*1.5
		mask = np.zeros((128,128), dtype="uint8")

		col_min = np.maximum(int(center[0]-face_scale), 0)
		col_max = np.minimum(int(center[0]+face_scale), 127)
		row_min = np.maximum(int(center[1]-face_scale), 0)
		row_max = np.minimum(int(center[1]+face_scale), 127)

		mask[row_min:row_max,col_min:col_max] = 1
		return kp, mask

def get_new_feature(X, feature):
	new_features = []
	for x_row, f_row in zip(X, feature):
		kp, mask = generate_kp(f_row)
		sift_vec = get_sift(x_row, kp = kp)
		hist_vec = get_sift(x_row, mode = "color", mask = mask)
		# concatnate them to be the new feature.
		new_features.append( np.append(sift_vec, hist_vec) )
	return new_features
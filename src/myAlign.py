import numpy as np
import cv2
from math import *

align = cv2.createAlignMTB()

def getBitMap(img):
	ret_tb = np.zeros(img.shape, np.uint8)
	ret_eb = np.zeros(img.shape, np.uint8)

	#get median of the img
	hist = cv2.calcHist(img, [0], None, [256], [0, 256])
	median, s = 0, 0
	threshold = img.size//2
	while(s < threshold and median < 256):
		s += int(hist[median])
		median += 1
	ret_tb = cv2.compare(img, median, cv2.CMP_GT)
	ret_eb = cv2.compare(np.abs(img-median), 4, cv2.CMP_GT)
	return ret_tb, ret_eb

def downSample(img):
	rows, cols = img.shape
	ret = np.zeros((rows//2, cols//2), np.uint8)
	offset = cols*2
	for i in range(rows//2):
		for j in range(cols//2):
			ret[i][j] = img[i*2][j*2]
	return ret

def imagePyramid(img, max_level):
	ret = [img]
	for level in range(max_level):
		ret.append(downSample(ret[level]))
	return ret

def calculateShift(img0, img1):
	rows, cols = img0.shape
	max_level = int(log(max(rows, cols))/log(2.))-1
	max_level = min(max_level, 5)
	p0 = imagePyramid(img0, max_level)
	p1 = imagePyramid(img1, max_level)

	shift = np.array([0, 0])
	for level in range(max_level, -1, -1):
		shift *= 2
		tb0, eb0 = align.computeBitmaps(p0[level])
		tb1, eb1 = align.computeBitmaps(p1[level])

		min_err = int(p0[level].size)
		cur_shift = np.array(shift)
		for i in range(-1, 2):
			for j in range(-1, 2):
				 new_shift = shift + np.array([i, j])
				 ts = (new_shift[0], new_shift[1])
				 shift_tb1 = align.shiftMat(tb1, ts)
				 shift_eb1 = align.shiftMat(eb1, ts)
				 temp = np.bitwise_xor(tb0, shift_tb1)
				 temp = np.bitwise_and(temp, eb0)
				 temp = np.bitwise_and(temp, shift_eb1)
				 if (np.count_nonzero(temp) < min_err):
				 	cur_shift = new_shift
				 	min_err = np.count_nonzero(temp)

		shift = cur_shift


	return shift



def imageAlign(images):
	pivot = len(images)//2
	ret = [None]*len(images)
	ret[pivot] = images[pivot]

	gray = cv2.cvtColor(images[pivot], cv2.COLOR_BGR2GRAY)
	shift = np.array([])

	for i in range(len(images)):
		if i == pivot:
			shift = np.append(shift, np.array([0, 0]))
		else:
			g = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
			temp_shift = calculateShift(gray, g)
			shift = np.append(shift, temp_shift)
			ts = (temp_shift[0], temp_shift[1])
			ret[i] = align.shiftMat(images[i], ts)
	return ret
import numpy as np
import cv2
import sys
from math import *

def process(image):
	ret = np.zeros(image.shape, np.float32)
	min_val = np.min(image)
	max_val = np.max(image)
	if (max_val-min_val > sys.float_info.epsilon):
		ret = np.true_divide(image-min_val, max_val-min_val)
	else:
		ret = image

	return ret

def get_bias(image):
	over_exposure = (np.asarray(image)>255).sum()/3
	under_exposure = (np.asarray(image)<.1).sum()/3
	rows,cols,_ = image.shape
	image_size = rows*cols
	over_exposure = min(0.15, 0.15*over_exposure/(image_size/2))
	under_exposure = min(0.25, 0.35*under_exposure/image_size)
	print("Tone mapping bias determined:", str(.85 + over_exposure - under_exposure))
	return .85 + over_exposure - under_exposure

def get_saturation(image):
	return .7

def mapLuminance(image, luminance, new_luminance, saturation):
	channels = cv2.split(image)
	for bgr in range(3):
		channels[bgr] = np.true_divide(channels[bgr], luminance)
		channels[bgr] = np.power(channels[bgr], saturation)
		channels[bgr] = np.multiply(channels[bgr], new_luminance)
	return cv2.merge(channels)

def initiateDragoToneMapping(image):
	bias = get_bias(image)
	ret = np.zeros(image.shape, np.float32)
	img = process(image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	log_gray = np.log(gray)
	mean = exp(np.mean(log_gray))
	gray = np.true_divide(gray, mean)

	min_val, max_val, _,_ = cv2.minMaxLoc(gray)
	adaptive_map = np.log(gray+1)
	adaptive_denom = np.power(np.true_divide(gray, max_val), log(bias)/log(.5))
	adaptive_denom = np.log(8.*adaptive_denom+2)
	adaptive_map = np.true_divide(adaptive_map, adaptive_denom)

	ret = mapLuminance(image, gray, adaptive_map, get_saturation(image))
	return process(ret)




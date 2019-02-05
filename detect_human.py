import numpy as np
from PIL import Image
from math import floor,log
import pickle
import imutils
from HOG import *
import cv2


def detect_windows(image_path, weights, bias, threshold_score,window_size,scale_step,slide):
	height_of_window = window_size[0]
	width_of_window = window_size[1]
	windows = {}
	scores = []
	im = np.array(Image.open(image_path).convert('L')).astype('float')
	im_height = im.shape[0]
	im_width = im.shape[1]
	end_scale = min(im_height/height_of_window, im_width/width_of_window)
	levels_in_scale_space = int(floor(log(end_scale)/log(scale_step) + 1))
	for l in range(0,levels_in_scale_space):
		resize_width = int(round(im_width/(scale_step**l)))
		resize_height = int(round(im_height/(scale_step**l)))
		if(l!=0):
			im_resized = imutils.resize(im_resized, width=resize_width, height=resize_height)
		else:
			im_resized = im.copy()
		print('level in scale space:',l)
		print('resized image shape:',np.shape(im_resized))
		resize_height = im_resized.shape[0]
		resize_width = im_resized.shape[1]
		v_margin = (resize_height - height_of_window) % slide
		h_margin = (resize_width - width_of_window) % slide
		v_top_left_corner = int(floor(v_margin / 2))
		h_top_left_corner = int(floor(h_margin / 2))
		v_bottom_right_corner = int(resize_height - v_margin + v_top_left_corner - 1)
		h_bottom_right_corner = int(resize_width - h_margin + h_top_left_corner - 1)
		im_cropped = im_resized[v_top_left_corner:v_bottom_right_corner+1, h_top_left_corner:h_bottom_right_corner+1]
		feature_vector = HOG(im_cropped, cell_size = 8, block_size=2, stride=1, no_of_bins = 9, binning_type='unsigned')
		rows = feature_vector.shape[0]
		columns = feature_vector.shape[1]
		virtual_window_size = [540,7]
		v_slide = 36
		h_slide = 1
		last_index_v = rows - virtual_window_size[0]
		last_index_h = columns - virtual_window_size[1]
		for m in range(0,last_index_v+1,v_slide):
			for n in range(0,last_index_h+1,h_slide):
				features_X = feature_vector[m:m+540,n:n+7]
				X = flatten_HOG(features_X,block_size=2,no_of_bins=9)
				score = np.dot(X,weights) + bias
				if(score>threshold_score):
					scores.append(score)
					x1 = m/v_slide * slide + v_top_left_corner
					y1 = n/h_slide * slide + h_top_left_corner
					x2 = x1 + height_of_window - 1 + v_top_left_corner
					y2 = y1 + width_of_window - 1 + h_top_left_corner
					bounding_box = [x1,y1,x2,y2]
					if l in windows:
						windows[l].append(bounding_box)
					else:
						windows[l] = [bounding_box]
	return windows,scores

def scale_windows(windows,scale_step):
	scaled_windows = []
	for level,boxes in windows.items():
		scale = scale_step**(level)
		if(level==0):
			for bounding_box in boxes:
				bounding_box[0] = int(round(bounding_box[0]* scale))
				bounding_box[1] = int(round(bounding_box[1]* scale))
				bounding_box[2] = int(round(bounding_box[2]* scale))
				bounding_box[3] = int(round(bounding_box[3]* scale))
				scaled_windows.append(bounding_box)
			continue
		for bounding_box in boxes:
			bounding_box[0] = int(round(bounding_box[0]* scale))
			bounding_box[1] = int(round(bounding_box[1]* scale))
			bounding_box[2] = int(round(bounding_box[2]* scale))
			bounding_box[3] = int(round(bounding_box[3]* scale))
			scaled_windows.append(bounding_box)
	return scaled_windows

def non_maxima_suppression(windows,scores):
	max_score_windows = []
	idx = sorted(range(len(scores)), reverse=True, key=lambda index:scores[index])
	no_of_windows = len(scores)
	final_windows = [1]*no_of_windows
	for i in range(0,no_of_windows):
		# if(final_windows[idx[i]]!=0):
		window = windows[idx[i]]
		centre = [window[0]+(window[2] - window[0])/2, window[1]+(window[3]-window[1])/2]
		for j in range(i+1,no_of_windows):
			if(final_windows[idx[j]]!=0):
				w = windows[idx[j]]
				if(centre[0]>w[0] and centre[0]<w[2] and centre[1]>w[1] and centre[1]<w[3]):
					final_windows[idx[j]] = 0
	for i in range(0,no_of_windows):
		if(final_windows[i]==1):
			window = windows[i]
			window[0] = window[0]+8
			window[1] = window[1]+8
			window[2] = window[2]-8
			window[3] = window[3]-8
			max_score_windows.append(windows[i])
	return max_score_windows



def display_image(image_path,windows):
	im = cv2.imread(image_path)
	for rect in windows:
		cv2.rectangle(im,(rect[1],rect[0]),(rect[3],rect[2]),(0,255,0),2)
	cv2.imshow('People detection',im)
	cv2.imwrite('DetectedImages/people5.png',im)
	cv2.waitKey(0)
	# if k==27:
	# 	cv2.destroyAllWindows()
	# elif k==ord('s'):
	# 	cv2.imwrite('DetectedImages/sample.png')
	# 	cv2.destroyAllWindows()


def detect_people(image_path):
	window_size = [128, 64]
	with open('SVM_models/SVM_retrained2.pickle','rb') as handle:
		clf = pickle.load(handle)
	weights = np.squeeze(clf.coef_)
	bias = clf.intercept_
	windows,scores = detect_windows(image_path, weights , bias,threshold_score = 0,window_size=window_size,scale_step = 1.2, slide = 8)
	scaled_windows = scale_windows(windows,scale_step=1.2)
	nms_windows = non_maxima_suppression(scaled_windows,scores)
	print(len(scores))
	print(nms_windows)
	display_image(image_path,nms_windows)



def main():
	#image_path = 'chinmay.jpeg'
	image_path = 'INRIAPerson/Test/pos/person_207.png'
	detect_people(image_path)
	

if __name__ == '__main__':
	main()
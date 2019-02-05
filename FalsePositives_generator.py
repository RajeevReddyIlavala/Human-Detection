import numpy as np
from PIL import Image
import os
import imutils
from math import floor,log
from HOG import *
import pickle
import time


def generate_hard_examples(neg_images_path, weights, bias, threshold_score,window_size,scale_step,slide):
	height_of_window = window_size[0]
	width_of_window = window_size[1]
	false_positives = []
	count = 1
	for dirpath, dirnames, filenames in os.walk(neg_images_path):
		for f_name in filenames:
			if(count>10):
				break
			print("mining image no :",count)
			count+=1
			start2 = time.time()
			image_path = os.path.join(dirpath,f_name)
			im = np.array(Image.open(image_path).convert('L')).astype('float')
			print('image size:',np.shape(im))
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
				print('cropped image size:',np.shape(im_cropped))
				feature_vector = HOG(im_cropped, cell_size = 8, block_size=2, stride=1, no_of_bins = 9, binning_type='unsigned')
				print('size of feature_vector:',np.shape(feature_vector))
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
							false_positives.append(X)
			end2 = time.time()
			runtime2 = end2-start2
			print("time taken:",runtime2)
	false_positives = np.array(false_positives)
	return false_positives




def main():
	start1 = time.time()
	neg_images_path = "INRIAPerson/train_64x128_H96/neg"
	window_size = [128, 64]
	with open('SVM_models/SVM_retrained.pickle','rb') as handle:
		clf = pickle.load(handle)
	weights = clf.coef_
	bias = clf.intercept_
	weights = np.squeeze(weights)
	false_positives = generate_hard_examples(neg_images_path, weights , bias,threshold_score = 0.2,window_size=window_size,scale_step = 1.2, slide = 8)
	train_neg_labels = np.full((false_positives.shape[0]), -1)
	# false_positives_dataset = [false_positives,train_neg_labels]
	# with open('TrainingDataHOG/false_positives.pickle','wb') as handle:
	# 	pickle.dump(false_positives_dataset,handle,protocol=pickle.HIGHEST_PROTOCOL)
	end1 = time.time()
	runtime1 = end1 - start1
	print('runtime:', runtime1)
	print(np.shape(false_positives))
	print(np.shape(train_neg_labels))



if __name__ == '__main__':
	main()
from HOG import *
import os
from PIL import Image
import numpy as np
import pickle
import random
import time

def extract_pos_features(pos_path):
	pos_HOG_features = []
	for dirpath, dirnames, filenames in os.walk(pos_path):
		for f_name in filenames:
			image_path = os.path.join(dirpath,f_name)
			im = np.array(Image.open(image_path).convert('L')).astype('float')
			im = im[16:im.shape[0]-16,16:im.shape[1]-16]
			feature_vector = HOG(im, cell_size = 8, block_size=2, stride=1, no_of_bins = 9, binning_type='unsigned')
			pos_HOG_features.append(feature_vector)
	return pos_HOG_features

def extract_neg_features(neg_path,window_size,windows_per_image):
	rows_per_window = window_size[0]
	columns_per_window = window_size[1]
	neg_HOG_features = []
	for dirpath, dirnames, filenames in os.walk(neg_path):
		for f_name in filenames:
			image_path = os.path.join(dirpath,f_name)
			im = np.array(Image.open(image_path).convert('L')).astype('float')
			no_of_rows = im.shape[0]
			no_of_columns = im.shape[1]
			max_index_rows = no_of_rows - rows_per_window
			max_index_columns = no_of_columns - columns_per_window
			row_step = ceil(max_index_rows/windows_per_image)
			column_step = ceil(max_index_columns/windows_per_image)
			for i in range(0,windows_per_image):
				m = int(random.randrange(0,max_index_rows,row_step))
				n = int(random.randrange(0,max_index_columns,column_step))
				window_image = im[m:m+rows_per_window, n:n+columns_per_window]
				feature_vector = HOG(window_image, cell_size = 8, block_size=2, stride=1, no_of_bins = 9, binning_type='unsigned')
				neg_HOG_features.append(feature_vector)
	return neg_HOG_features





def main():
	start = time.time()
	pos_path = "INRIAPerson/train_64x128_H96/pos"
	neg_path = "INRIAPerson/train_64x128_H96/neg"
	pos_HOG_features = extract_pos_features(pos_path)
	neg_HOG_features = extract_neg_features(neg_path,[128,64],10)
	pos_HOG_features = np.asarray(pos_HOG_features)
	neg_HOG_features = np.asarray(neg_HOG_features)
	pos_labels = np.full((pos_HOG_features.shape[0]),1)
	neg_labels = np.full((neg_HOG_features.shape[0]),-1)
	train_x = np.vstack([pos_HOG_features,neg_HOG_features])
	train_y = np.hstack([pos_labels,neg_labels])
	dataset_HOG = [train_x,train_y]
	with open('TrainingDataHOG/dataset_HOG.pickle','wb') as handle:
		pickle.dump(dataset_HOG,handle,protocol=pickle.HIGHEST_PROTOCOL)
	end = time.time()
	run_time = end - start
	print(run_time)

if __name__ == '__main__':
	main()



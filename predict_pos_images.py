import os
import numpy as np
from PIL import Image
from HOG import *
import pickle
from sklearn import svm
import random

def extract_pos_features_test(pos_path):
	no_of_test_images = 0
	pos_HOG_features = []
	for dirpath, dirnames, filenames in os.walk(pos_path):
		for f_name in filenames:
			no_of_test_images += 1
			image_path = os.path.join(dirpath,f_name)
			im = np.array(Image.open(image_path).convert('L')).astype('float')
			im = im[3:im.shape[0]-3,3:im.shape[1]-3]
			feature_vector = HOG(im, cell_size = 8, block_size=2, stride=1, no_of_bins = 9, binning_type='unsigned')
			features = flatten_HOG(feature_vector,2,9)
			pos_HOG_features.append(features)
	print(no_of_test_images)
	return pos_HOG_features

def extract_neg_features(neg_path,window_size,windows_per_image):
	no_of_test_images = 0
	rows_per_window = window_size[0]
	columns_per_window = window_size[1]
	neg_HOG_features = []
	for dirpath, dirnames, filenames in os.walk(neg_path):
		for f_name in filenames:
			no_of_test_images +=1
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
				features = flatten_HOG(feature_vector,2,9)
				neg_HOG_features.append(features)
	print('no of test images:',no_of_test_images)
	return neg_HOG_features




def test_SVM(x):
	# with open('SVM_models/SVM_preliminary.pickle','rb') as handle1:
	# 	clf_preliminary = pickle.load(handle1)
	with open('SVM_models/SVM_retrained2.pickle','rb') as handle2:
		clf_retrained = pickle.load(handle2)
	# labels_preliminary = clf_preliminary.predict(x)
	weights = np.squeeze(clf_retrained.coef_)
	bias = clf_retrained.intercept_
	labels_retrained = x@np.expand_dims(weights,axis = 1)+bias
	labels_retrained[labels_retrained>0] = 1
	labels_retrained[labels_retrained<=0] = -1
	print(np.shape(labels_retrained))
	# labels_retrained = clf_retrained.predict(x)
	# accuracy_preliminary = abs(np.sum(labels_preliminary[labels_preliminary==1])/len(labels_preliminary))
	accuracy_retrained = abs(np.sum(labels_retrained[labels_retrained==-1])/len(labels_retrained))
	print(accuracy_retrained)



def main():
	# pos_test_path = 'INRIAPerson/test_64x128_H96/pos'
	# test_pos_x = extract_pos_features_test(pos_test_path)
	# test_pos_x = np.asarray(test_pos_x)
	# test_features_pos = [test_pos_x]
	# with open('TestingDataHOG/test_pos.pickle','wb') as handle:
	# 	pickle.dump(test_features_pos,handle,protocol=pickle.HIGHEST_PROTOCOL)
	# neg_test_path = 'INRIAPerson/test_64x128_H96/neg'
	# test_neg_x = extract_neg_features(neg_test_path, [128,64], 10)
	# test_neg_x = np.asarray(test_neg_x)
	# test_features_neg = [test_neg_x]
	# with open('TestingDataHOG/test_neg.pickle','wb') as handle:
	# 	pickle.dump(test_features_neg,handle,protocol=pickle.HIGHEST_PROTOCOL)
 	# with open('TestingDataHOG/test_pos.pickle','rb') as handle:
 	# 	test_features_pos = pickle.load(handle)
 	# test_pos_x=test_features_pos[0]
 	with open('TestingDataHOG/test_neg.pickle','rb') as handle:
 		test_features_neg = pickle.load(handle)
 	test_neg_x=test_features_neg[0]
 	print(np.shape(test_neg_x))
 	print(type(test_neg_x))
 	test_SVM(test_neg_x)


if __name__ == '__main__':
	main()




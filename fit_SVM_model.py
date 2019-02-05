import pickle
import numpy as np
from sklearn import svm
import time
import random

def train_SVM(train_x,train_y):
	clf = svm.SVC(C = 0.01, kernel = 'linear')
	svm_model = clf.fit(train_x,train_y)
	return svm_model

def main():
	with open('TrainingDataHOG/dataset_HOG.pickle','rb') as handle1:
		dataset_HOG = pickle.load(handle1)
	with open('TrainingDataHOG/false_positives.pickle','rb') as handle2:
		false_positives = pickle.load(handle2)
	train_x = dataset_HOG[0]
	train_y = dataset_HOG[1]
	false_positives_x = false_positives[0]
	false_positives_y = false_positives[1]
	no_of_false_positives = false_positives_x.shape[0]
	#randomly selecting --start
	random_positions = random.sample(range(0,no_of_false_positives), 1200)
	false_positives_x = false_positives_x[random_positions,:]
	false_positives_y = false_positives_y[random_positions]
	#randomly selecting --end
	print('initial train data size:',np.shape(train_x))
	print('false_positives train data set size:', np.shape(false_positives_x))
	print(np.shape(train_y),np.shape(false_positives_y))
	train_features = np.vstack([train_x,false_positives_x])
	print('shape of final train set:',np.shape(train_features))
	train_labels = np.hstack([train_y,false_positives_y])
	print('shape of final train labels:',np.shape(train_labels))
	start = time.time()
	svm_model = train_SVM(train_features,train_labels)
	end = time.time()
	train_time = end-start
	print(train_time)
	with open('SVM_models/SVM_retrained2_1200.pickle','wb') as handle:
		pickle.dump(svm_model,handle,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()





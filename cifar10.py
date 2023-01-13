from collections import Counter
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import time
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

def print_same_line(string):
	sys.stdout.write('\r' + string)
	sys.stdout.flush()


"""
CIFAR-10 Dataset: "Learning Multiple Layers of Features from Tiny Images" 
Alex Krizhevsky, 2009.
"""
class CIFAR10:
	def __init__(self, data_path):
		"""Extrait les données CIFAR10 depuis data_path"""
		file_names = ['data_batch_%d' % i for i in range(1,6)]
		file_names.append('test_batch')

		X = []
		y = []
		for file_name in file_names:
			with open(data_path + file_name, 'rb') as fin:
				data_dict = pickle.load(fin, encoding='bytes')
			X.append(data_dict[b'data'].ravel())
			y = y + data_dict[b'labels']

		self.X = np.asarray(X).reshape(60000, 32*32*3)
		self.y = np.asarray(y)

		fin = open(data_path + 'batches.meta', 'rb')
		self.LABEL_NAMES = pickle.load(fin, encoding='bytes')[b'label_names']
		fin.close()

	def train_test_split(self):
		X_train = self.X[:50000]
		y_train = self.y[:50000]
		X_test = self.X[50000:]
		y_test = self.y[50000:]

		return X_train, y_train, X_test, y_test

	def all_data(self):
		return self.X, self.y

	def __prep_img(self, idx):
		img = self.X[idx].reshape(3,32,32).transpose(1,2,0).astype(np.uint8)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img

	def show_img(self, idx):
		cv2.imshow(self.LABEL_NAMES[self.y[idx]], self.__prep_img(idx))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def show_examples(self):
		fig, axes = plt.subplots(5, 5)
		fig.tight_layout()
		for i in range(5):
			for j in range(5):
				rand = np.random.choice(range(self.X.shape[0]))
				axes[i][j].set_axis_off()
				axes[i][j].imshow(self.__prep_img(rand))
				axes[i][j].set_title(self.LABEL_NAMES[self.y[rand]].decode('utf-8'))
		plt.show()	
		

dataset = CIFAR10('./cifar-10-batches-py/')
X_train, y_train, X_test, y_test = dataset.train_test_split()
X, y = dataset.all_data()

dataset.show_examples()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from collections import Counter
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import time
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
#importation du K plus proche voisin 
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

#Algo du plus proche voisin        
class NearstNeighbor :
    
    
    #ON souhaite comparee chaque exemple de test a tout les exemples d'entrainements
    
    # Collecte tout les parametres ou hyperparametres comme la fonction distance des images
    
    def __init__(self, distance_func='l1'):
        self.distance_func = distance_func
    
    # Recupere les donnees d'entrainements 
    def train(self, X, y):
        """ X est une matrice N x D dans laquelle chaque ligne est un exemple de test.
        y est une matrice N x 1 de valeurs correctes"""
        self.X_tr = X.astype(np.float32)
        self.y_tr = y
        
    # Compare tout les exemples de test avec tout les exemples d'entrainement avec la dimention D
    
    def predict(self, X):
        
        """ X est une matrice M x D dans laquelle chaque ligne est un exemple de test."""
        """ Fait la prediction pour tout les exemples de test et calcul la precision de la prediction """
        """ Donc une image sera compare a toutes les autres images et nous retourne une valeur de prediction qui correspond 
            a une classe de 0 a 9 (etiquettes)"""
        """ On donne X (test) et la fonction retourne Y (prediction)"""
        
        #Convertion en float 32 bytes en cas de valeurs negatives
        
        X_te = X.astype(np.float32)
        
        #recupere le nombre d'exemple de tests
        
        num_test_examples = X.shape[0]
        
        """prediction qui doit avoir le meme nombres de lignes que notre matrice d'entree X"""
            
        #On souhaite obtenir le meme type de donnes que y_tr de la fonction train 
        #on initialise y_pred une matrice contenant autant de zeros au'il y a d'exemples test et de meme type de donne que Y_tr
        y_pred = np.zeros(num_test_examples, self.y_tr.dtype)
        
        #Trouve l'image d'entrainement dont la distance avec l'image de test est la plus petite possible
        for i in range(num_test_examples):
            if self.distance_func == 'l2':
                # matrice de test - matrice d'entrainement
                #Cette soustraction nous retourne une matrice col de resultat de N lignes
                #Dans cette col on recupere l'exemple d'entrainement qui a la plus petite valeur
                #avec valeur au carre 
                distances = np.sum(np.square(self.X_tr - X_te[i]), axis = 1)
            else:
                #avec valeur absolue
                distances = np.sum(np.abs(self.X_tr - X_te[i]), axis = 1)
            
            #Retoune l'index de la plus petite valeur de distance 
            smallest_dist_idx = np.argmin(distances)
            #On recupere son index(son etiquette) et la transferer a la prediction y
            y_pred[i] = self.y_tr[smallest_dist_idx]
                
        return y_pred       
                
        
    
		

dataset = CIFAR10('./cifar-10-batches-py/')
X_train, y_train, X_test, y_test = dataset.train_test_split()
X, y = dataset.all_data()

dataset.show_examples()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

'''nn = NearstNeighbor()
nn.train(X_train, y_train)
y_pred = nn.predict(X_test[:100])

#Compare la prediction au valeurs reel y_test afinde calculer le presition de l'algo avec le nombres de valeurs correct sur le total de valeur
#avec une methode numpy
accuracy = np.mean(y_test[:100]  == y_pred)
print(accuracy)
'''
#  KNeighborsClassifier(nombre de plus proche voisin 'donc K', distance 'l1 ou l2', nombres de coeurs du processeur utilise ici on les utilise tousavec -1 )
knn = KNeighborsClassifier(n_neighbors = 5, p = 1, n_jobs = -1)
#Entrainement du classificateur avec les donnees d'entrainement X et Y
knn.fit(X_train, y_train)
#Prediction sur 100 parametres
y_pred = knn.predict(X_test[:100])
#Verification de la precision du model
#on compare les valeur et on divise par le total
accuracy = np.mean(y_test[:100] == y_pred)
#affichage du resultat
print(accuracy)

#Variation des hyperparametres pour connaitre la valeur optimal pour K et la meilleur fonction de distance L

#grille de recherche
#param_grid = {valeur de K[],la fonction de distance[l1 ou l2 ]}
param_grid = {'n_neighbors': [1,3,5,10,20,50,100],'p': [1,2]}
#GridSearchCV(knn, param_grid,5 partition pour la validation croisee, nombres de coeurs du processeur)
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1)
#j'appel ma fonction fit , il effectue la validation croisee sur les 60 000 img et fractionner ca en 5 partitions
grid_search.fit(X_train,y_train)
#affiche les meilleurs parametres
print(grid_search.best_params_)

#-*-coding: utf-8 -*-

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from sklearn import grid_search
import numpy as np
import MLP_dropOut
reload(MLP_dropOut)

import numpy as np
import pylab as pl
mnist=fetch_mldata('MNIST original')
Xmnist = mnist.data
Xmnist = Xmnist/255.0 #on met toutes les valeurs entre 0 et 1
Ymnist = mnist.target
encod = OneHotEncoder()
Y = Ymnist.reshape(Ymnist.shape[0], 1)
encod.fit(Y)
Y = encod.transform(Y).toarray()
iperm = np.arange(Xmnist.shape[0])
np.random.shuffle(iperm)
X = Xmnist[iperm]
Y = Y[iperm]
ending_index = 20000
Xtrain = X[0:ending_index]
Ytrain = Y[0:ending_index]
mlp = MLP_dropOut.MLP([784,100,10], learning_rate=0.35, n_iter=10, auto_update_lr=True)
mlp.fit(Xtrain, Ytrain)

print "Score apprentissage de {} % avec un échantillon d'apprentissage comptant {} elements".format(mlp.score(X[0:ending_index],Y[0:ending_index]), ending_index)


#il faut mettre à jour les poids (*0,1 si dropOut avec une proba de 0,1)!!

#il faut aussi mettre un drop out sur la couche d entree
print "Score generalisation de {} % ".format(mlp.score(X[ending_index:],Y[ending_index:]))
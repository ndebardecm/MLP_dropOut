# -*-coding: utf-8 -*-

"""Autoencoder"""

import numpy as np
import pylab as pl
import copy as cp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score


def sigmo(x):
    return 1.0/ (1+np.exp(- x))

def sigmoid(x):
    return 1.5* sigmo(x) -0.25

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.5 * sigmo(x) * (1-sigmo(x))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x) ** 2

def relu(x):
    #rectified linear unit = detecteur de pattern = nul en dessous d'un seuil
    y = x
    y[x < 0] = 0  # annule les valeurs de x negatives
    return y

def drelu(x):
    y = x
    y[x < 0] = 0
    y[x > 0] = 1


class Layer:
    def __init__(self, ninputs, noutputs, learning_rate=0.01, wdecay=0.0, momentum=0.0, drop_out = True):
        #momentum = mise en memoire de la valeur des poids de la couche precedente pour lisser les poids d'une couche à l'autre
        self.ninputs = ninputs  #dimension en entree de la couche
        self.noutputs = noutputs  #dimension en sortie de la couche
        self.learning_rate = learning_rate
        self.wdecay = wdecay
        self.momentum = momentum

        #initilaisation des poids
        self.W = np.zeros((self.noutputs, self.ninputs))
        self.Wbias = np.zeros(
            self.noutputs)  #Poids appliques au entrees correspiondant aux biais. Servent à faire converger les poids inutiles vers 0.
        self.dW = np.zeros((self.noutputs, self.ninputs))
        self.dWbias = np.zeros(self.noutputs)
        self.fa = sigmoid  #fonction d'activation
        self.dfa = dsigmoid
        self.statesIn = np.zeros(self.ninputs)
        self.statesOut = np.zeros(self.noutputs)
        self.activOut = np.zeros((self.noutputs,1))
        self.deltas = np.zeros(self.noutputs)
        self.drop_out = drop_out

    def forward(self, inputs):
        self.statesIn = inputs  #on conserve l'etat d'entree pour la retropropagation
        if(self.drop_out == True):
            #On inhibe aleatoirement des neurones, soit l entree de la couche
            n_coord_to_drop_out = np.random.randint(0,int(0.1*self.ninputs))
            if n_coord_to_drop_out > 0:
                for i in range(0,n_coord_to_drop_out):
                    self.statesIn[np.random.randint(0,n_coord_to_drop_out)] = 0
                    #print "Coordonnees mises a 0 : ", n_coord_to_drop_out
        self.activOut = np.dot(self.W, inputs) + self.Wbias
        self.statesOut = self.fa(self.activOut)
        return self.statesOut

    def backward(self, err):
        self.delta = err * self.dfa(self.activOut)
        err = np.dot(self.W.T, self.delta)
        return err


    def compute_gradient_step(self):
        self.dW = self.learning_rate * np.dot(self.delta.reshape(self.delta.shape[0], 1), self.statesIn.reshape(1, self.statesIn.shape[0])) + 2 * self.wdecay * self.W
        self.dWbias = self.learning_rate * self.delta

    def gradient_step(self):
        self.W = self.W - self.dW
        self.Wbias = self.Wbias - self.dWbias

    def reset(self):
        self.W = np.random.normal(0, 1 / np.power(self.ninputs, 0.5) * np.power(self.noutputs, 0.5),
                                  (self.noutputs, self.ninputs))
        self.Wbias = np.random.normal(0, 1 / np.power(self.ninputs, 0.5) * np.power(self.noutputs, 0.5),
                                      (self.noutputs))
        #on remet les autres a zero
        self.dW = np.zeros((self.noutputs, self.ninputs))
        self.dWbias = np.zeros(self.noutputs)
        self.statesIn = np.zeros(self.ninputs)
        self.statesOut = np.zeros(self.noutputs)


class MLP:
    def __init__(self, SpecLayers=[784, 100, 10], learning_rate=0.01, wdecay=0.0, momentum=0.0, n_iter=10, auto_update_lr=True):
        self.SpecLayers = SpecLayers
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate
        self.wdecay = wdecay
        self.momentum = momentum
        self.n_iter = n_iter
        self.layers = []
        self.auto_update_lr = auto_update_lr
        for i in range(1, len(self.SpecLayers)):
            self.layers.append(
                Layer(self.SpecLayers[i - 1], self.SpecLayers[i], self.learning_rate, self.wdecay, self.momentum))
        self.reset()

    def set_params(self, **parameters):
        #fonction necessaire pour GridSearch
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        #il faut recreer le reseau de neurones
        self.layers = []
        for i in range(1, len(self.SpecLayers)):
            self.layers.append(
                Layer(self.SpecLayers[i - 1], self.SpecLayers[i], self.learning_rate, self.wdecay, self.momentum))

        #mise à jour du learning rate et du weight decay sur toutes les couches
        self.set_learning_rate(self.learning_rate)
        self.set_wdecay(self.wdecay)
        self.reset()

    def set_learning_rate(self, lr):
        for l in self.layers:
            l.learning_rate = lr

    def set_wdecay(self, wdecay):
        for l in self.layers:
            l.wdecay = wdecay

    def propagate_forward(self, data):
        s = self.layers[0].forward(data)
        for i in range(1, len(self.layers)):
            s = self.layers[i].forward(s)
        return s

    def fit(self, X, Y):
        # Y est de dimension particuliere: plutôt qu'une seule colonne où la valeur correspond à la classe, il y a autant de colonnes que de classes possibles. Pour savoir à quelle ligne appartient un exemple (une ligne), on regarde la colonne où il y a un 1 (ex: col 5 <=> classe 5)
        # Pour les tests, on regarde la colonne où la valeur est max
        n_examples = X.shape[0]
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.9, 0.95, 1.05, 1.5, 2, 10]
        #learning_rates = [0.35]
        errold = 1000000.0
        freq = 1
        freq_ref = 3
        for iteration in range(self.n_iter):
            err = 0.0
            for i in range(n_examples):
                err += self.gradient_step(X[i], Y[i])
                #print "\n err : ", err /n_examples
            err /= n_examples
            print "\n erreur quadratique: ", err
            if self.auto_update_lr == False:
                #Regle de readaptaion du pas - on diminue le pas de gradient...
                self.learning_rate = self.learning_rate*0.1
                #self.learning_rate = self.learning_rate/(1 + self.learning_rate_init*iteration)
            else:
                if freq == freq_ref:
                    freq = 1
                    #on met à jour le learning_rate suivant les cas:
                    best_lr = self.learning_rate
                    best_err = err #initialise a err car si on ne peut pas mieux faire il ne sert a rien de changer le pas de gradient
                    for lr in learning_rates:
                        temp_mlp = cp.deepcopy(self)
                        Xtemp = X[0:0.1 * X.shape[0], :]
                        Ytemp = Y[0:0.1 * Y.shape[0], :]
                        temp_err = temp_mlp.fit_partial(Xtemp, Ytemp, lr)
                        #print "temp_err: {}, learning rate: {}".format(temp_err, lr)
                        if(temp_err < best_err):
                            best_err = temp_err
                            best_lr = lr
                    print "\n Best learning rate : {} for iteration {}".format(best_lr, iteration+1)
                    self.learning_rate = best_lr
                else:
                    freq += 1

            #on met à jour sur les couches
            self.set_learning_rate(self.learning_rate)
            if(np.abs(err - errold)/errold < 0.0001): #on arrete d'iterer si on est suffisamment proche du minimum
                break
            errold = err
        print "\n Erreur quadratique finale: ", err
        return self

    def fit_partial(self, X, Y, lr):
        self.set_learning_rate(lr)
        n_examples = X.shape[0]
        err = 0.0
        for i in range(n_examples):
            err += self.gradient_step(X[i], Y[i])
        return err / n_examples

    def predict_classe(self, X):
        s = self.propagate_forward(X)
        return np.argmax(s)  #on cherche la colonne pour laquelle la valeur obtenue est max, ce qui donne la classe

    def predict_TS(self, X):
        n_examples = X.shape[0]
        ypredit = np.zeros((n_examples, 1))
        for i in range(n_examples):
            ypredit[i] = self.predict_classe(X[i])
        return ypredit

    def score(self, X, Y):
        # necessaire pour faire un grid search
        ypredit = self.predict_TS(X)
        yreal = np.argmax(Y, axis=1)
        return accuracy_score(ypredit, yreal)

    def gradient_step(self, inputData, desiredOutput):
        s = self.propagate_forward(inputData)
        err = self.compute_loss(s, desiredOutput)
        self.propagate_backward(desiredOutput)
        self.maj_weights()
        return err

    def compute_loss(self, predict, desired):
        #critere sur l'erreur quadratique, pas dur la classification
        return np.linalg.norm(predict - desired, 2) / len(predict)

    #le deuxieme argument precise la norme à utiliser

    def propagate_backward(self, desired):
        prediction = self.layers[-1].statesOut  #etat de sortie de la derniere couche
        err = prediction - desired
        for i in range(0, len(self.layers)):
            err = self.layers[len(self.layers) - i - 1].backward(err)

    def maj_weights(self):
        for i in range(0, len(self.layers)):
            self.layers[len(self.layers) - i - 1].compute_gradient_step()
            self.layers[len(self.layers) - i - 1].gradient_step()

    def reset(self):
        for l in self.layers:
            l.reset()

    def get_params(self,deep=True):
        return {'SpecLayers': self.SpecLayers,
                'learning_rate': self.learning_rate,
                'wdecay': self.wdecay,
                'momentum': self.momentum,
                'n_iter': self.n_iter}


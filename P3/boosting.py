import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################

		#print(len(features))
		
		pred = np.zeros(len(features))

		for clf,beta in zip(self.clfs_picked, self.betas):
			pred+= beta*np.array(clf.predict(features))

		pred[pred <= 0]=-1
		pred[pred > 0]=1

		return pred.tolist()
		
		
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		
		N = len(labels)

		w = np.ones(N)/N
		

		for t in range(self.T):

			eps = float('inf')

			for clf in self.clfs:

				pred_clf = clf.predict(features)

				clf_error = np.sum(w * (np.array(labels)!= np.array(pred_clf)) )

				if clf_error < eps:
					ht= clf
					eps= clf_error
					htx = pred_clf

			self.clfs_picked.append(ht)

			beta = 1/2*np.log((1-eps)/eps)
			self.betas.append(beta)

			for n in range(N):
				if htx[n]==labels[n]:
					w[n]*= np.exp(-beta)

				else:
					w[n]*= np.exp(beta)

			w /= np.sum(w)
		

		




		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	
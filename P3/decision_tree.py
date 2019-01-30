import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			
			branches= np.array(branches)


			class_totals = np.sum(branches,axis=0)

			entropy = branches/class_totals

			branch_fractions = class_totals/np.sum(class_totals)

			entropy = np.sum(np.array([[-i*np.log2(i) if i>0 else 0 for i in j ]for j in entropy]), axis=0)

			entropy = np.sum(entropy* branch_fractions)

			return entropy


		
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			



			min_entropy = float('inf')

			x= np.array(self.features)[:, idx_dim]



			consider_feature = True
			for i in x:
				if i==None:
					consider_feature= False

			if not consider_feature:
				continue

			unique_branch_values = np.unique(x)

			branches = np.zeros((self.num_cls, len(unique_branch_values)))

			for i, branch_val in enumerate(unique_branch_values):
				y= np.array(self.labels)[np.where(x==branch_val)]

				for yx in y:
					branches[yx, i]+=1

			entropy = conditional_entropy(branches)

			if entropy < min_entropy:
				min_entropy=entropy
				self.feature_uniq_split= unique_branch_values.tolist()
				self.dim_split= idx_dim
			


		############################################################
		# TODO: split the node, add child nodes
		############################################################

	
		x_i = np.array(self.features)[:, self.dim_split]

		x= np.array(self.features, dtype=object)

		x[:, self.dim_split] = None

		for val in self.feature_uniq_split:
			indices = np.where(x_i == val)

			x_n = x[indices]
			x_n = x_n.tolist()

			y_n = np.array(self.labels)[indices]
			y_n=y_n.tolist()

			child = TreeNode(x_n, y_n, self.num_cls)

			if np.array(x_n).size ==0:
				child.splittable= False

			#none_cnt = [1 if split_val is None for split_val in x_n[0]]

			if all(v is None for v in x_n[0]):
				child.splittable= False
			
			self.children.append(child)

		
		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()
		
		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max




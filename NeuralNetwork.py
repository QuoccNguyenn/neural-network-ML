import numpy as np
np.random.seed(2)

def sigmoid(x):#ham kich hoat Sigmoid Function
	return 1 / (1 + np.exp(-x))

def dSigmoid(y):#Dao ham cua sigmoid
	return y * (1 - y)

def map(func, x):
	for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				x[i][j] = func(x[i][j])
	return	x

def addElement(x, y):#ham cong phan tu trong ma tran
	if(x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]):
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				x[i][j] += y[i][j]
		return	x
	else:
		return

def subtractElement(x,y):#ham tru cac phan tu trOng ma tran
	if(x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]):
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				x[i][j] = (x[i][j] - y[i][j])
		return x
	else:
		return

def multiply(X,n):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i][j] *= n
	return	X
	
def multyplyHadamard(x,y):
	if(x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]):
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				x[i][j] *= y[i][j]
		return	x
	else:
		return

class NeuralNetwork():

	def __init__(self,input_nodes,layer1_nodes,layer2_nodes,layer3_nodes,output_nodes):
		self.input_nodes = input_nodes
		self.layer1_nodes = layer1_nodes
		self.layer2_nodes = layer2_nodes
		self.layer3_nodes = layer3_nodes
		self.output_nodes = output_nodes

		self.w_input_layer1 = np.random.rand(layer1_nodes, input_nodes) *2 -1 #[-1; 1]  W_I_H (2,2) [-1;1]
		self.w_layer1_layer2 = np.random.rand(layer2_nodes, layer1_nodes) *2 -1
		self.w_layer2_layer3 = np.random.rand(layer3_nodes, layer2_nodes) *2 -1
		self.w_layer3_output = np.random.rand(output_nodes, layer3_nodes) *2 -1

		self.bias_layer1 = np.random.rand(layer1_nodes,1) *2 -1
		self.bias_layer2 = np.random.rand(layer2_nodes,1) *2 -1
		self.bias_layer3 = np.random.rand(layer3_nodes,1) *2 -1
		self.bias_output = np.random.rand(output_nodes,1) *2 -1

		self.learning_rate = 0.2

	def feedforward(self,inputs):
		
		inputs = np.array([inputs]).T 					#bien doi thanh ma tran va chuyen vi inputs
		#tinh toan o hidden layer:
		#Layer 1
		layer1 = np.dot(self.w_input_layer1, inputs)	#nhan ma tran inputs cho ma tran weights cua hidden layer
		layer1 = addElement(layer1, self.bias_layer1)	#cong vao gia tri bias
		layer1 = map(sigmoid, layer1)					#dung ham kich hoat

		#Layer 2
		layer2 = np.dot(self.w_layer1_layer2, layer1)
		layer2 = addElement(layer2, self.bias_layer2)
		layer2 = map(sigmoid, layer2)

		#Layer 3
		layer3 = np.dot(self.w_layer2_layer3, layer2)
		layer3 = addElement(layer3,self.bias_layer3)
		layer3 = map(sigmoid, layer3)

		#tinh toan ouput
		output = np.dot(self.w_layer3_output, layer3)	#nhan gia tri cua hidden layer 3 cho weights cua output
		output = addElement(output, self.bias_output)	#cong vao gia tri bias
		output = map(sigmoid, output)					#dung ham kich hoat cho output

		return layer1, layer2, layer3, output

	def predict(self, inputs):
		layer1,layer2,layer3,outputs = self.feedforward(inputs)
		return outputs

	def train(self,inputs,targets):#ham tinh toan va lay sai so BACKPROPAGATION
		layer1, layer2, layer3, outputs = self.feedforward(inputs)

		inputs = np.array([inputs]).T

		targets = np.array([targets]).T#gia tri tham chieu

		#gia tri Error cua output da tinh va ket qua thuc te E = 
		output_errors = subtractElement(targets,outputs)
		##

		#tinh Gradient output
		output_gradient = map(dSigmoid, outputs)
		output_gradient = multyplyHadamard(output_gradient,output_errors)
		output_gradient = multiply(output_gradient, self.learning_rate)
		##

		#chuyen vi ma tran trong so cua layer2 va output
		delta_W_l3o = np.dot(output_gradient,layer3.T)
		##

		#cap nhat w va bias tuong ung
		self.w_layer3_output = addElement(self.w_layer3_output, delta_W_l3o)
		self.bias_output = addElement(self.bias_output, output_gradient)
		####

		#layer 3
		w_layer3_output_trans = (self.w_layer3_output).T

		#tinh sai so error
		layer3_errors = np.dot(w_layer3_output_trans, output_errors)

		#tinh gradient
		layer3_gradient = map(dSigmoid,layer3)
		layer3_gradient = multyplyHadamard(layer3_gradient, layer3_errors)
		layer3_gradient = multiply(layer3_gradient, self.learning_rate)
		delta_W_l23 = np.dot(layer3_gradient, layer2.T)
		#cap nhat trong so va bias
		self.w_layer2_layer3 = addElement(self.w_layer2_layer3, delta_W_l23)
		self.bias_layer3 = addElement(self.bias_layer3, layer3_gradient)

		#tuong tu voi layer 2
		w_layer2_layer3_trans = (self.w_layer2_layer3).T

		layer2_errors = np.dot(w_layer2_layer3_trans, layer3_errors)
		
		layer2_gradient = map(dSigmoid,layer2)
		layer2_gradient = multyplyHadamard(layer2_gradient, layer2_errors)
		layer2_gradient = multiply(layer2_gradient, self.learning_rate)
		delta_W_l12 = np.dot(layer2_gradient, layer1.T)

		self.w_layer1_layer2 = addElement(self.w_layer1_layer2, delta_W_l12)
		self.bias_layer2 = addElement(self.bias_layer2, layer2_gradient)

		#tuong tu voi layer1
		w_layer1_layer2_trans = (self.w_layer1_layer2).T
		layer1_errors = np.dot(w_layer1_layer2_trans, layer2_errors)

		layer1_gradient = map(dSigmoid,layer1)
		layer1_gradient = multyplyHadamard(layer1_gradient, layer1_errors)
		layer1_gradient = multiply(layer1_gradient, self.learning_rate)
		delta_W_il1 = np.dot(layer1_gradient, inputs.T)

		self.w_input_layer1 = addElement(self.w_input_layer1, delta_W_il1)
		self.bias_layer1 = addElement(self.bias_layer1, layer1_gradient)
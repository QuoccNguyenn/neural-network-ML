import numpy as np
from NeuralNetwork import NeuralNetwork

training_data = [ # p | q > r
	{
		'inputs': [0,0,0],
		'target': [0.]
	},
	{
		'inputs': [0,0,1],
		'target': [1.]
	},
	{
		'inputs': [0,1,0],
		'target': [0.]
	},
	{
		'inputs': [0,1,1],
		'target': [1.]
	},
	{
		'inputs': [1,0,0],
		'target': [0.]
	},
	{
		'inputs': [1,0,1],									#
		'target': [1.]										#
	},
	{
		'inputs': [1,1,0],
		'target': [0.]
	},
	{
		'inputs': [1,1,1],
		'target': [0.]
	}
]

nn= NeuralNetwork(3,4,2,2,1)

print("Untrained Neural Network:\n")
for i in range(8):
	data = training_data[i]
	results = nn.predict(data['inputs'])
	print("results: ",round(results[0,0],3))


print("\nTrained Neural Network:\n")

for i in range (10000):
	r = np.random.randint(8)
	data = training_data[r]

	nn.train(data['inputs'], data['target'])


for i in range(8):
	data = training_data[i]
	results = nn.predict(data['inputs'])
	print("result: ",round(results[0,0],3), " -> ",data['target'])
'''
Excercise done by following sentdex
Neural Networks from Scratch in Python
https://nnfs.io
www.pythonprogramming.net
'''

# Imports relevan libraries
import numpy as np # numpy for linear algebra operations
import nnfs # neural networks from scratch library
from nnfs.datasets import spiral_data # Gets data for neural network


# Define layer density 
class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	# Forward pass of neural network
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

# Defines a rectified linear function
class Activation_ReLU:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

# Defines a binary activation function
class Activation_Binary:
	def forward(self, inputs):
		self.output = np.maximum(0,(np.minimum(1, inputs)))


# Defines main script as a function
def main():

	# Access data from nnfs datasets
	X, y = spiral_data(100, 3)

	# n_inputs is 2 beacuse of 2 dimentional vector space that defines each data point
	layer1 = Layer_Dense(2,4)
	
	# creates an empty object
	activation1 = Activation_Binary()

	layer1.forward(X)
	activation1.forward(layer1.output)

	print(activation1.output)


if __name__ == '__main__':
	main()

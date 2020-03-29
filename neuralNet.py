import numpy as np
import matplotlib.pyplot as plt

class Layer:
	def __init__(self, size):
		self.input = np.random.rand(size).reshape(-1, 1)
		self.output = np.append(np.zeros(1), self.input).reshape(-1, 1)

class NeuralNet:
	def __init__(self, shape):
		self.layers = [Layer(size) for size in shape]
		self.weights = [np.random.rand(shape[i + 1], shape[i] + 1) for i in range(len(shape) - 1)]
		self.numClasses = shape[-1]

	def feedForward(self, x):
		self.layers[0].output = np.append(np.ones(x.shape[0]).reshape(-1, 1), x, axis=1)
		for i in range(len(self.weights)):
			self.layers[i + 1].input = np.dot(self.layers[i].output, self.weights[i].T)
			output = NeuralNet.activation(self.layers[i + 1].input)
			self.layers[i + 1].output = np.append(np.ones(output.shape[0]).reshape(-1, 1), output, axis=1)
			if i == (len(self.weights) - 1):
				self.layers[i + 1].output = output

	def predict(self, x):
		self.feedForward(x)
		return np.argmax(self.layers[-1].output, axis=1)

	def loss(self, x, y, reg=3):
		numExamples = x.shape[0]
		self.feedForward(x)
		predicted = self.layers[-1].output
		sum1 = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted)) / numExamples

		sum2 = 0
		for weight in self.weights:
			sum2 += np.sum(weight[:, 1:] ** 2)
		sum2 *= (reg / (2 * numExamples))
		return sum1 + sum2

	def activation(input):
		return 1 / (1 + np.exp(-input))

	def activationGradient(input):
		return NeuralNet.activation(input) * (1 - NeuralNet.activation(input))

	def update(self, x, y, learnRate, reg=3):
		numExamples = x.shape[0]
		self.feedForward(x)
		predicted = self.layers[-1].output
		for i in reversed(range(len(self.weights))):
			if i == (len(self.weights) - 1):
				error = predicted - y
			else:
				error = np.dot(error, self.weights[i + 1][:, 1:]) * NeuralNet.activationGradient(self.layers[i + 1].input)
			gradient = np.tensordot(error, self.layers[i].output, axes=([0], [0])) / numExamples
			gradient += (reg / numExamples) * np.append(np.zeros(self.weights[i].shape[0]).reshape(-1, 1), self.weights[i][:, 1:], axis=1)
			self.weights[i] -= learnRate * gradient

	def train(self, x, y, epochs, learnRate):
		lossVals = [self.loss(x, y)]
		for i in range(epochs):
				self.update(x, y, learnRate)
				lossVals.append(self.loss(x, y))
		return lossVals

	def loadWeights(self, files):
		for i in range(len(files)):
			self.weights[i] = np.loadtxt(files[i], delimiter=',')

if __name__ == "__main__":
	numClasses = 10
	network = NeuralNet([400, 25, numClasses])
	x, y = np.loadtxt('./data/X.csv', delimiter=','), np.loadtxt('./data/Y.csv', delimiter=',', dtype=np.int)
	y = np.eye(numClasses)[y - 1]
	learnRate = 0.2
	epochs = 500
	network.loadWeights(['./data/initial_W1.csv', './data/initial_W2.csv'])
	loss = network.train(x, y, epochs, learnRate)

	count, total = 0, len(y)
	prediction = np.eye(numClasses)[network.predict(x)]
	for i in range(len(x)):
		if np.array_equal(prediction[i], y[i]):
			count += 1
	accuracy = count / total
	print("Accuracy of network: ", accuracy)
	toBePredicted = [2171, 145, 1582, 2446, 3393, 815, 1378, 529, 3945, 4628]
	toBePredicted = [x-1 for x in toBePredicted]

	print("Predicted labels: ", (network.predict(x)[toBePredicted] + 1) % 10)
	print("Actual labels: ", (np.argmax(y[toBePredicted], axis=1) + 1) % 10)

	plt.plot(range(epochs + 1), loss)
	plt.title("Loss vs. Iterations")
	plt.ylabel("Loss")
	plt.xlabel("Iterations")
	plt.show()

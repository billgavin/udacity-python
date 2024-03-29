
# coding: utf-8

# In[6]:

from numpy import exp, array, random, matmul

class NeuralNetwork():
    
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = matmul(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment
            
    def think(self, inputs):
        return self.__sigmoid(matmul(inputs, self.synaptic_weights))


# In[7]:

if __name__ == "__main__":
    nn = NeuralNetwork()
    print "Random starting synaptic weights:"
    print nn.synaptic_weights
    
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    nn.train(training_set_inputs, training_set_outputs, 10000)
    
    print "New synaptic weights after training:"
    print nn.synaptic_weights
    print "Considering new situation [1,0,0] -> ? :"
    print nn.think(array([1,0,0]))


# In[ ]:




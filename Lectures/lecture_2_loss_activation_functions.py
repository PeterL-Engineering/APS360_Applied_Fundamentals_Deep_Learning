'''
Artifical Neuron
- Deep learning is inspired by the human brain
- We need to study the building blocks of the brain to understand deep learning

- Neurons receive throuhg dendrites, process information in the cell body, and send signals through axons
- We cam represent a neuron as a mathematical function
- The neuron receives inputs (x1, x2, x3, ...) and produces an output (y)
- The neuron has weights (w1, w2, w3, ...) that determine the importance of each input
- The neuron has a bias (b) that determines the threshold for activation
- The full equation for a neuron is: (y) = f(w1*x1 + w2*x2 + w3*x3 + ... + b)
- The activation function (f) determines the output of the neuron based on the weighted sum of inputs
- Most datasets are not linearly separable, so we need to use non-linear activation functions

Example: Price of a House
- X = [Year, Square Footage, Number of Rooms, Number Built]
    - [2001, 3000, 5, 1] and y = [1.5 Million]
- Y = Price of the house
- If f is the identity function, the neuron will output the weighted sum of inputs:
- y = w1*2001 + w2*3000 + w3*5 + w4*1 + b

Types of Activation Functions
- Early activation functions were either step or sign functions
- These functions are not differentiable, so we cannot use gradient descent to optimize the weights
- Sigmoid function is a smooth, S-shaped curve that maps inputs to outputs between 0 and 1 eg. tanh
- The issue with sigmoid is that it saturates for large inputs, causing the gradient to be very small
- ReLU (Rectified Linear Unit) is a popular activation function that is non-linear and does not saturate
    - ReLU(x) = max(0, x)
    - Leaky ReLU is a variant of ReLU that allows a small gradient for negative inputs
    - SiLU (Sigmoid Linear Unit) is another variant that is similar to ReLU but has a small slope for negative inputs
    - SoftPlus is a smooth approximation of ReLU that is differentiable everywhere

Training Neural Networks
- Given input x, predicted output y, ground truth t, and Neuron M
- Make a prediction for some input data with a known correct ouptut t: y = M(x; w)
- Compare the prediction to the ground truth using a loss function L(y, t)
- Adjust the weights of the neuron to minimize the loss function using gradient descent
- Repeat until we have an acceptable loss L / error E

- At some point, the loss function will begin to increase for validation data
- This is called overfitting, and it means the model is too complex and is memorizing the training data

Example: Labelling an Image
- Given a 2x2 image with 4 pixels of a cat we want to label it as such using a model
- The model flattens the image into a 4x1 vector and passes it through neurons
- In this case there are 3 neurons so there is a 3x4 matrix of weights
- The model outputs a 3x1 vector of probabilities for each class (cat, dog, bird)
- The model uses a softmax activation function to convert the output into probabilities (logits)
- One-hot encoding is used to represent the classes as binary vectors -> compare with softmax output

Types of Loss Functions
- Mean Squared Error (MSE) is used for regression tasks (The difference between the predicted and actual values)
- Cross Entropy (CE) is used for classification tasks (The difference between the predicted and actual probabilities)
    - We determine the surprise by -ground_truth(x) * log(probability(x))
    - This is reduced to -log(probability(x)) for one-hot encoded classes
    - The more surprised we are, the more we need to adjust the weights
- Binary Cross Entropy (BCE) is a special case of cross entropy for binary classification tasks
'''
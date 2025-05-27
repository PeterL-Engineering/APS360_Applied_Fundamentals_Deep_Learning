'''
Training a Neuron
    - How do we update the weights of a neuron given the loss function?
    - We use gradient descent to minimize the loss function
    - First we need to know the gradient of the loss function with respect to the weights
    - We can update the weights via: w_ji = w_ji - learning_rate * dL/dw_ji
        - The learning rate determines how much we adjust the weights
        - The derivative is multiplied by negative one to minimize instead of maximize the loss

Importance of Multiple Layers
    - An easy example of the importance of multiple layers is the XOR problem
    - The XOR problem is not linearly separable, so a single layer cannot solve it
    - To increase efficiency of updating weights in a deep network, we use backpropagation
    - Backpropagration requires us to constantly recompute the gradients with respect to the weights
    - It is important to use dynamic programming to avoid recomputing the same gradients multiple times
    - The goal of these layers is to project the data into a higher dimensional space where it is linearly separable

Tricks to Train Deep Networks
    - Hyperparamter Tunings
        - Some of these hyperparameters include batch size, learning rate, number of epochs, and number of layers
        - Batch size is the number of samples used in one iteration of training
        - Learning rate is the step size used in gradient descent
        - Number of epochs is the number of times the entire dataset is used in training
        - Number of layers is the number of hidden layers in the network
        - We tune these hyperparameters via random search since it is computationally expensive to train a deep network

    - Stochastic Gradient Descent (SGD)
        - SGD is a variant of gradient descent that updates the weights after each sample
        - This allows the model to learn faster and avoid local minima
        - It is important to shuffle the data before training to avoid bias in the order of samples
        - We use mini-batch gradient descent to balance the benefits of SGD and batch gradient descent
        - Using momentum helps to accelerate SGD in the relevant direction and dampens oscillations
        - The new formula: v_ji = momentum * v_ji - learning_rate * dL/dw_ji and w_ji = w_ji + v_ji

    - Adaptive Moment Estimate (Adam)
        - Adam is an optimization algorithm that combines the benefits of SGD and momentum
        - It adapts the learning rate for each weight based on the first and second moments of the gradients
        - The formula for Adam is: m_ji = beta1 * m_ji + (1 - beta1) * dL/dw_ji and v_ji = beta2 * v_ji + (1 - beta2) * (dL/dw_ji)^2
        - The weights are updated using: w_ji = w_ji - learning_rate * m_ji / (sqrt(v_ji) + epsilon)
        - Adam is widely used in deep learning due to its efficiency and effectiveness

    - Learning Rate
        - The learning rate is a hyperparameter that determines the step size in gradient descent
        - A small learning rate can lead to slow convergence, while a large learning rate can lead to overshooting the minimum
        - Learning rate scheduling is a technique to adjust the learning rate during training

    - Normalization
        - We normalize the inputs to prevent the model from being biased towards certain features
        - Normalization techniques include min-max scaling, z-score normalization, and batch normalization
        - Calculated via batch mean and variance, and then normalized via: x' = (x - mean) / sqrt(variance + epsilon)
        - Inference time normalization is done via running mean and variance
        - Normalization helps to stabilize the training process and improve convergence

'''
'''
Hyperparameter Tuning Continued
    - Layer Normalization
        - Layer normalization normalizes the inputs across the features instead of the batch
        
Regularization
    - Regularization is a technique to prevent overfitting in deep networks
    
    - Dropout
        - Dropout forces an NN to learn more robust features (makes task more difficult)
        - During training drop activations (set to zero) with probability p
            - Eg. if p = 0.5, then any layer activation w/ probability greater than 0.5 is set to zero
        - During inference: multiply weights by (1 - p) to keep same distribution
            - No longer want task to be difficult, so we keep all activations
    
    - Weight Decay (L2 Regularization)
        - Prevents the weights from growing too much (lowering variance)
        - W_t+1 = W_t - learning_rate*(alpha * W_t + dE/dW_t)

    - Early Stopping w/ Patience
        - As soon as validation loss starts to increase, start a counter
        - If validation loss does not decrease for a certain number of epochs (patience), stop training

Introduction to Pytorch
    - If you are using nn.Module you do not need to define the backward pass
    - Typically use BCEWithLogitsLoss for binary classification since it combines a sigmoid layer and the binary cross entropy loss in one class
    - For multi-class classification, use CrossEntropyLoss which combines softmax and cross entropy loss
    - Forward pass is defined as model(input) which makes the prediction
    - Backward pass is defined as loss.backward() which computes the gradients
    - Optimizer is defined as optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Debugging Neural Networks
    - Make sure your model can overfit the training data
    - Make sure that your network is training (ie loss is decreasing)
    - Double check variables names and various other programming errrors
    - Confusion matrix (True positive (TP), True negative (TN), False positive (FP), False negative (FN))
        - When the dataset is imbalanced we use precision (TP / (TP + FP)) and recall (TP / (TP + FN)) to evaluate the model
    - 2D projections of Data (PCA, t-SNE)

Convolutional Neural Networks (CNNs)
    - Covolution is a mathematical operation that combines two functions to produce a third
    - Since images are at least 2D, we expand the convolution operation to 2D
        - Convlution of image I with kerel K is defined as:
        - (I * K)(x, y) = sum_{i, j} I(x + i, y + j) * K(i, j)
    - CNNs learn filters that are convolved with the input image to produce feature maps
    - Typical blueprint for model: Data -> Encoder -> Classifer -> Output
    - Zero padding is used to keep the spatial dimensions of the input and output the same
    - Stride is the number of pixels by which the filter is moved across the input image
    VERY IMPORTANT: Output size: o = (i - k + 2p) / s + 1
#     - Where i is the input size, k is the kernel size, p is the padding, and s is the stride
      - If it is non square then you must calculate the output size for each dimension separately
    
    CNNs on RGB Inputs
        - The kernel becomes a 3D tensor with dimensions (kernel_height, kernel_width, input_channels)
        - Ex. Colour input (3x28x28) with kernel (3x3x3)
            - How many trainable weights are there? 3 * 3 * 3 = 27
        - To extract more features, we can use multiple kernels in parallel
        - Thus if we had 10 kernels, the output would be (10x26x26) if we used a stride of 1 and no padding
    '''
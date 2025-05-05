''' 
Machine Learning Basics
 - Supervised Learning
    - Regression (real-valed or continuous) or classification (categorical eg. Goat detection)
    - Requires data with ground-truth labels
 - Unsupervised Learning
    - Self-supervised and semi-supervised learning
    - Requires observations without human annotations
 - Reinforcement Learning
    - Learning from interaction with the environment
    - Actions affects the environment (like training a dog)

More on Supervised Learning
- Model makes a prediction, and then the prediction is compared to the ground-truth label
- The difference between the prediction and the ground-truth label is called the loss or error
- The model then updates its parameters to minimize the loss
- The process of updating the model's parameters is called training
- Inference is the process of making predictions on new data using a trained model

Polynomial Regression
- Determine the coefficients such that predictions are as close to data
- If the model memorizes the data, it is overfitting (vice versa)
- We can use regularization to prevent overfitting
- Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function
- The penalty term discourages the model from fitting the noise in the data

Train/Validation/Test Split
- Train set: used to train the model
- Validation set: used to tune the model's hyperparameters and evaluate its performance
- Test set: used to evaluate the model's performance on unseen data
- If you only have one dataset, randomly split it into train and test sets
- If you have eg cats, dogs, and goats, make sure that split is same in all splits (30% cats in train, test, and validation sets)
'''
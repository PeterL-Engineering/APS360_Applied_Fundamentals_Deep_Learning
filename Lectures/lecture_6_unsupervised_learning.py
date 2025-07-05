'''Unsupervised Learning

    - Motivation
        - Humans learn via patterns without explicit supervisory signals
        - Feature clustering is the equivalent of human learning
        - Can we make the model cluster the data into groups without labels?

    - Autoencoders
        - Encoders convert the inputs into internal representations via dimensionality reduction
        - Decoders convert the internal representations back into the original input space
        - Instead of classifying, we reconstruct the input data
        - The model is forced to learn the most important features of the data because of the bottleneck
        - Can be used for anomaly detection by comparing the reconstruction error (Autoencoder bad at reconstructing anomalies)
        - Also good for feature extraction, generating new data, and data compression

        - Denoising Autoencoders
            - Adding noise to the input data helps to regularize and prevent overfitting
            - Gaussian noise is a simple and effective way to add noise
            - If your model is able to ignore the noise then that means it has a deeper understanding of the content

        - Generating New Images
            - The network can save space by mapping similar images to the same internal representation
            - The decoder can then generate new images by sampling from the internal representation space
            - Eg. take two embeddings and interpolate between them to create a new image
            - When plotting interpolated images, we can group similar images together in embedding space

        Variational Autoencoders (VAEs)
            - Probabilistic -> their outputs are partly determined by chance even after training
            - Generative -> they can generate new instances that look like they were sampled from the training set
            - Encoder generates a normal distribution (two vectors corresponding to mean and deviation) instead of a fixed embedding
            - To allow the VAE to sample from this distribution, we use the reparameterization trick
            - A sample is taken from a separate normal distribution and then scaled and shifted by the mean and deviation vectors
            - This allows the VAE to use gradient descent to optimize the parameters of the distribution
            - Instead of learning the small regions, the VAE learns the entire distribution of the data

            - KL Divergence is used to measure the difference between the two distributions (the learned distribution and the prior distribution)
            - It is basically cross-entropy between the two distributions
            - If we plug in the encoder distributions and the prior into KL divergence of two multivariate Gaussians:
                D_KL(p|q) = 0.5 *sum(mu^2 + sigma^2 - log(sigma^2) - 1)
            - In total we have two losses: one from the reconstruction and one from the KL divergence (prioritize reconstruction loss)

    - Convolutional Autoencoders
        - Encoders learn visual embedding using convolutional layers
        - Decoders up=sample the learned visual embedding using transposed convolutional layers
            - Applying kernel to all pixels in the input image then add overlapping regions in output (instead of sliding the kernel)
            - Padding is inverse ie. we remove border from output image
            - Output padding is used to ensure that the output size matches the input size
        - Pre-trained Autoencoders
            - Instead of removing classifier, just remove the decoder part of the autoencoder

Self-Supervised Learning

    - Motivation
        - What if we define proxy supervised tasks such that the model defines its own labels
        - Contrastive learning is a popular self-supervised learning technique
        - The model compares the two images in embedding space instead of comparing the images themselves

    - SimCLR
        - A self-supervised learning framework that uses contrastive learning
        - The model learns to maximize the similarity between two augmented views of the same image
        - It uses a projection head to map the embeddings to a lower-dimensional space
        - The model is trained using a contrastive loss function (NT-Xent loss)
        - The model learns to distinguish between positive pairs (augmented views of the same image) and negative pairs (augmented views of different images)
        - Note that the model will push away two embeddings even if the two embeddings were of the same dog
        
'''
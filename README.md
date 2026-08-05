# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Arjit Katiyar's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/arjit_108.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement AdaDelta Update Step | Implement a vectorized AdaDelta update in NumPy using running gradient and parameter-update averages without a manual learning rate. | https://www.tensortonic.com/problems/adadelta-optimizer |
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Anchor Box Generation | Generate object-detection anchor boxes across a feature grid for every scale and aspect-ratio combination. | https://www.tensortonic.com/problems/anchor-box-generation |
| Batch Normalization (Forward) | Implement the batch-normalization forward pass in NumPy using feature-wise statistics, scale, shift, and numerical stability. | https://www.tensortonic.com/problems/batch-normalization |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Implement Contrastive Loss (Siamese) | Implement Siamese-network contrastive loss using pair labels, embedding distances, and a separation margin. | https://www.tensortonic.com/problems/contrastive-loss |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Implement Dice Loss | Compute Dice loss for segmentation predictions using overlap, total mass, and a numerical smoothing term. | https://www.tensortonic.com/problems/dice-loss |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Focal Loss | Compute mean binary focal loss from predicted probabilities using a configurable focusing parameter. | https://www.tensortonic.com/problems/focal-loss |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Gradient Clipping (Global Norm) | Clip a NumPy gradient array by its global L2 norm while preserving direction when scaling is required. | https://www.tensortonic.com/problems/gradient-clipping |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Learning Rate Scheduler (Linear Decay) | Compute a linearly decaying learning rate across training steps between configured start and end values. | https://www.tensortonic.com/problems/linear-lr-scheduler |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Implement Nadam (Nesterov + Adam) | Implement one Nadam optimizer step in NumPy by combining Adam moments with Nesterov momentum. | https://www.tensortonic.com/problems/nadam-optimizer |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Implement R² Score (Coefficient of Determination) | Compute the coefficient of determination from targets and predictions with explicit constant-target handling. | https://www.tensortonic.com/problems/r2-score |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Compute Silhouette Score | Compute the mean silhouette score from intra-cluster and nearest-cluster distances for labeled samples. | https://www.tensortonic.com/problems/silhouette-score |
| Implement a Simple CNN Layer (NumPy) | Implement a NumPy CNN layer forward pass with batched valid convolution across channels and bias addition. | https://www.tensortonic.com/problems/simple-cnn-layer |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| Implement TF-IDF Vectorizer | Build TF-IDF document vectors from token counts and inverse document frequency across a text corpus. | https://www.tensortonic.com/problems/tfidf-vectorizer |
| Implement Triplet Loss | Compute triplet loss from anchor, positive, and negative embeddings using distances and a margin. | https://www.tensortonic.com/problems/triplet-loss |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Implement Wasserstein Critic Loss | Compute Wasserstein critic loss as the difference between mean fake and real critic scores. | https://www.tensortonic.com/problems/wasserstein-critic-loss |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/arjit_108)
<!-- tensortonic:end -->

# This code is the source code implementation for the paper "HD-FL: A Federated Learning Privacy Preservation Model Based on Homomorphic Encryption".



## Abstract 

![image-20240617020352934](/Users/wangjinning/Library/Application Support/typora-user-images/image-20240617020352934.png)

In recent years, federated learning has been widely used in the field of machine learning, and the privacy leakage problem has become more and more serious. The privacy-preserving techniques of federated learning are still immature. This paper proposes the HD-FL model for the privacy leakage problem of federated learning, which combines differential privacy with homomorphic encryption. Homomorphic encryption is used to encrypt the gradient parameters, and the global ciphertext is calculated to update the gradient parameters with differential privacy. HD-FL can prevent inference of the exchanged model information during the training process while ensuring that the generated models also have acceptable prediction accuracy. The global model update allows the client to prevent inference threats and generate highly accurate models without sacrificing privacy. Moreover, this model can be used to train various machine learning methods, which are experimentally validated with three different datasets, and the results show that this approach outperforms current solutions in terms of security and accuracy.



## Experimental Environment

**Operating environment：**

GPU：NVIDIA A100 SXM4 40G GPU 

CPU：Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz

**Installation：**

To run the code, you need to install the following packages：

```
numpy                  	1.16.2
tensorflow-privacy      0.5.1
pytorch                 1.4.0
torch                   1.7.1
syft                    0.2.9
scipy                   1.4.1
phe                     1.4.0
```

## Datasets:

```
MNIST
Fashion-MNIST
CIFAR-10
```

## Experimental Setup

**Hyperparameters:**

- Training is conducted over 300 rounds with 10, 50, and 100 clients participating in different experiments.
- The learning rate is set at 0.01, and the batch size is 32.
- The client model consists of a Convolutional Neural Network (CNN) with two convolutional layers, two max-pooling layers, and one fully connected layer.
- The activation function used is ReLU, and the optimizer employed is Gradient Descent.

**Privacy Mechanism:**

- **Differential Privacy:** Laplace noise with a privacy budget ϵ=10\epsilon = 10ϵ=10 is added to the gradient parameters on the client side before encryption.
- **Homomorphic Encryption**: Paillier homomorphic encryption algorithm is used to encrypt the gradient parameters before they are uploaded to the server.

**Training Process:**

1. **Initialization:** Clients initialize their local neural network models and generate gradient parameters using a random number algorithm.
2. **Encryption and Upload:** Gradient parameters are encrypted using the Paillier homomorphic encryption algorithm and uploaded to the server.
3. **Aggregation:** The server aggregates the encrypted gradients from all clients and updates the global model.
4. **Broadcast:** The updated global model is sent back to the clients for further training.

**Evaluation Metrics:**

- **Model Performance:** Model accuracy and loss are used as primary metrics to assess performance.
- **Privacy Protection:** The effectiveness of privacy protection is evaluated based on the ability of the model to maintain accuracy and reduce loss while incorporating privacy-preserving techniques.

## Files

#### `__init__.py`

This file sets up the HD-FL package structure, initializing the package and preparing it for use. It may contain metadata and necessary imports for the package.

#### `DPMechanisms.py`

This file implements differential privacy mechanisms crucial for the HD-FL model. It includes functions that add Laplace noise to gradient parameters, ensuring privacy during federated learning.

#### `FLModel.py`

This is the core file for the federated learning model. It handles client-side operations such as model training, gradient encryption, and communication with the server. The file integrates the homomorphic encryption and differential privacy methods to secure the learning process.

#### `MLModel.py`

This file defines the machine learning model architecture, specifically a Convolutional Neural Network (CNN) used for image classification. It outlines the layers and forward pass of the CNN.

#### `preprocess.py`

Responsible for data preprocessing, this file includes functions for loading datasets, normalizing data, and partitioning it across multiple clients. It ensures that the data is correctly formatted and distributed for federated learning.

#### `test.py`

This file contains test cases for verifying the functionality and performance of the HD-FL model. It ensures that the model's components work correctly and meet the expected standards.

#### `utils.py`

A collection of utility functions that support various operations in the project. These functions handle tasks such as encryption, decryption, and evaluation metrics, aiding the main processes.

#### `test.ipynb`

An interactive Jupyter notebook designed for running and visualizing tests on the HD-FL model. It is useful for debugging and demonstrating the capabilities of the model, providing a hands-on way to explore the project's features.

These explanations offer a concise overview of the critical code files in the HD-FL project, helping users quickly grasp their roles and importance within the project.



##  Experimental Results

The experimental results section of this paper focuses on evaluating the HD-FL model's performance using three datasets: MNIST, Fashion-MNIST, and CIFAR-10. The experiments are designed to test the accuracy and loss of the model compared to other existing models and to observe the impact of different numbers of clients on the model's performance.

![1](/Users/wangjinning/Desktop/2/李思雨-基于同态加密的联邦学习隐私保护模型/pic/1.png)

![2](/Users/wangjinning/Desktop/2/李思雨-基于同态加密的联邦学习隐私保护模型/pic/2.png)

![3](/Users/wangjinning/Desktop/2/李思雨-基于同态加密的联邦学习隐私保护模型/pic/3.png)


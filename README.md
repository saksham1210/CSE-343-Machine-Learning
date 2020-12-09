# CSE-343-Machine-Learning
Assignments of the course


## Assignment-1

### scratch.py

Implemented Linear and Logistic regressions from scratch using only NumPy.

### logisticSKlearn.py

Logistic regression implemented using Sklearn.

### SGD_Sklearn.py

SGD implementation using Sklearn

### EDA.txt

Conducted EDA on Bank Authentication Dataset (in datasets folder)

#

## Assignment-2

### Q1.py

Implemented SkLearn's PCA, SVD, Tsne. Did a stratified split and trained a logistic regression using Sklearn.

### Q2.py

Used Bootstrapping on the Linear regression model to measure bias and variance of the model.

### Q3.py

* Implemented Gridsearch and K-cross validation from scratch and used it on Sklearn's Decision-Trees and GNB classifier.
* Calculated Confusion matrix for binary and multiclass data and plotted their ROCs from scratch

### Q4.py

Implemented Gaussian Naive Bayes Classifier from scratch using only NumPy library

#

## Assignment-3

### Q1.py

Implemented Multilayer Perceptron with parameters: -
* **n layers**: Number of Layers (int)
* **layer sizes**: an array of size n layers which contains the number of nodes in each layer.
(array of int)
* **activation**: activation function to be used (string)
* **learning rate**: the learning rate to be used (float)
* **weight init**: initialization function to be used
  * **zero**: Zero initialization
  * **random**: Random initialization with a scaling factor of 0.01
  * **normal**: Normal(0,1) initialization with a scaling factor of 0.01
* **batch size**: batch size to be used (int)
* **num epochs**: number of epochs to be used (int)

### Q2.py

Used the MLP from Q1 on the MNIST dataset for various hyper-parameter settings

### Q3.py

Used PyTorch's MLP for a dataset.

### Q4.py

Used pre-trained Alexnet model from PyTorch.




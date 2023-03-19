# ANN-based Regressor in PyTorch for Regression-Base problem with Shapley explanations

## ANN-based Regressor in PyTorch for Regression-Base problem category as per Ready Tensor specifications.

### Tags

- ANN
- neural network
- PyTorch
- regression
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- fastAPI
- nginx
- uvicorn
- docker

### Introduction

This is an Simple ANN Regressor, with a single hidden layer with non-linear activation.
Model applies l1 and l2 regularization. This model is built using PyTorch.

Model explainability is provided using Shapley values. These explanations can be viewed by means of various plots.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for learning rate for the SDG optimizer, L1 and L2 regularization and the activation function for the neural network.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include PyTorch for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service.

The web service provides three endpoints-

- /ping for health check
- /infer for predictions in real time
- /explain for obtaining local explanations for few samples (maximum of 5)

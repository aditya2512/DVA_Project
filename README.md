# E-Commerce Store Demand Forecasting 

The E-commerce Store Demand Forecasting repository provides a comprehensive set of forecasting models and tools, enabling users to predict sales demand over a specific timeline by leveraging historical sales data. This accurate inventory forecasting strategy can be instrumental in optimizing stock levels, reducing costs, and enhancing customer service.

## Dataset 

This project makes use of the e-commerce events data from the cosmetics shop provided by [kaggle.com](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop/). Training data for this repository uses data from October 2019 to December 2019, with subsequent testing on data from January 2020 to February 2020.

## Models

Three Machine Learning models have been implemented:

1. **Random Forest** - A robust machine learning algorithm from the ensemble category, which is capable of processing multiple decision trees to improve forecasting accuracy.

2. **Linear Regression** - It simplifies the task of sales forecasting by fitting a relationship between the dependent and independent variables, producing an equation that predicts the variable of interest.

3. **BiLSTM (Bidirectional Long Short Term Memory)** - An innovative deep learning approach which utilizes recurrent neural networks to consider both past (backward) and future (forward) aspects for accurate forecasting.

## Structure of Repository 

- `notebooks/` - This directory contains all Jupyter notebooks used for the training of each of the three models. 

- `python_scripts/` - Here reside the Python scripts utilized in this forecasting project. 

- `run-` - These are bash scripts for executing Linear Regression, Random Forest, and BiLSTM models. These scripts automate the process of creating a conda environment, installing necessary dependencies from the `requirements.txt` file, and running the respective Python scripts. Additionally, these scripts output the trained models into the current directory. 

## How to Perform Model Training

To trigger the training process for the predictive models and build your own versions, follow the steps below:

1. **Download the Dataset** - Download the respective dataset and store it in the `data/` directory.
2. **Execute Training Scripts** - Depending on which model you want to train, execute the appropriate bash script:
    - For Bidirectional Long Short Term Memory (BiLSTM), run `run-bilstm.sh`
    - For Linear Regression, run `run-linear-regression.sh`
    - For Random Forest Regression, run `run-random-forest-regressor.sh`

## How to Replicate the Results

To make predictions with the provided trained models, you will need to follow these steps:

1. **Download the Model** − Download the respective model from the current directory.
2. **Preprocess Your Data** − Ensure to preprocess your data and perform the necessary feature engineering steps, the same way it was done with the training data.
3. **Make Predictions** − Use the provided model to make predictions.

## Dependencies 

Please refer to the `requirements.txt` file for a detailed list of dependencies needed to run each model. 

We hope that these resources will provide valuable insights into the process of demand forecasting for e-commerce.
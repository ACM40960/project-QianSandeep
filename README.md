# <span style="color:#2c3e50">Price Prediction of S&P 500</span>
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SciKit%20Learn-yellow.svg)
![Status](https://img.shields.io/badge/status-Completed-success.svg)

This project focuses on predicting the prices of the S&P 500 using different machine learning models. We experimented with three different models, each documented in separate Jupyter notebooks. The final model is presented in Model 3, which also includes detailed data analysis.

## <span style="color:#3498db">Table of Contents</span>
- [Price Prediction of S\&P 500](#price-prediction-of-sp-500)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Data](#data)
  - [Notebooks](#notebooks)
    - [Model 1](#model-1)
    - [Model 2](#model-2)
    - [Model 3 (Final Model)](#model-3-final-model)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Team](#team)

## <span style="color:#2ecc71">Project Overview</span>
The goal of this project is to predict the prices of the S&P 500 index. We experimented with different features and models to determine the best approach for accurate predictions. After thorough analysis, we finalized the model that performed the best in terms of prediction accuracy.

## <span style="color:#e67e22">Data</span>
The data used for this project consists of historical price information for the S&P 500 index. The dataset includes features such as:
- **Date**: The date of the recorded data.
- **Open**: The price of the S&P 500 at market open.
- **High**: The highest price of the S&P 500 on that day.
- **Low**: The lowest price of the S&P 500 on that day.
- **Close**: The closing price of the S&P 500.
- **Adj Close**: The adjusted closing price of the S&P 500, adjusted for dividends and splits.
- **Volume**: The number of shares traded.

The data was preprocessed to handle missing values and create new features that could potentially improve the prediction model. The data analysis and preprocessing steps are detailed in the notebooks.

Snippet of code of how the data is loaded in the notebooks:

```python
start = '2000-01-01'
end = datetime(2024, 8, 7, 11, 21, 24, 633194)

sp500_data = yf.download('^GSPC', start='2000-03-14', end='2024-01-01')
sp500 = sp500_data.dropna()
print(sp500.head(3))
```

For more details on the data collection and preprocessing, refer to the notebooks.

## <span style="color:#9b59b6">Notebooks</span>

### <span style="color:#e74c3c">Model 1</span>
- **Description**: This notebook focuses on using the 'Adj Close' feature for prediction.
- **Number of Code Cells**: 13

```python
# Example of model training in Model 1 using 'Adj Close' feature
from sklearn.linear_model import LinearRegression

# Selecting feature and target
X = df[['Adj Close']].values
y = df['Price'].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
```
[View Model 1 Notebook](./Project/Final%20folder/model1.ipynb)

### <span style="color:#f39c12">Model 2</span>
- **Description**: This notebook explores why predicting the 'Adj Close' price may not be the best approach.
- **Number of Code Cells**: 15

```python
# Example of feature engineering in Model 2
df['Price_Diff'] = df['Price'].diff()

# Selecting new features
X = df[['Volume', 'Price_Diff']].fillna(0).values
y = df['Price'].values

# Training the model with new features
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluating the model
r2_score = model.score(X_test, y_test)
print(f'R^2 Score: {r2_score}')
```

[View Model 2 Notebook](./Project/Final%20folder/model2.ipynb)

### <span style="color:#2980b9">Model 3 (Final Model)</span>
- **Description**: This is our final model, which includes comprehensive data analysis and the finalized prediction model.
- **Number of Code Cells**: 29

```python
# Example of hyperparameter tuning and final model training in Model 3
from sklearn.model_selection import GridSearchCV
 
 
# Defining parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Setting up the GridSearch with RandomForest
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearch
print(f'Best Parameters: {grid_search.best_params_}')

# Training final model with best parameters
final_model = RandomForestRegressor(**grid_search.best_params_)
final_model.fit(X_train, y_train)
```

[View Model 3 Notebook](./Project/Final%20folder/model3.ipynb)

## <span style="color:#27ae60">Installation</span>
To run the notebooks, you need to have Python and Jupyter installed. Additionally, install the required dependencies by running:

```bash
pip install -r requirements.txt
```


## <span style="color:#8e44ad">Usage</span>
Clone the repository and open the notebooks using Jupyter:


```bash
git clone https://github.com/your-repository/price-prediction-sp500.git
cd price-prediction-sp500
jupyter notebook
```

Run the notebooks in sequence, starting with Model 1, then Model 2, and finally Model 3, to understand the progression of the models and the final results.


## <span style="color:#27ae60">Team</span>
This project was developed by:

@QianqianM
@SandeepJabez


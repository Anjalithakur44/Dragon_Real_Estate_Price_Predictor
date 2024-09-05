## Project Overview
This project uses various features from a real estate dataset to predict house prices. The goal is to build a regression model that estimates property values based on crime rates, zoning classifications, number of rooms, and property age. Accurate price prediction can assist investors and analysts in making informed decisions in the real estate market.

## Dataset
The dataset used in this project is housing_data.csv, which includes the following features:
CRIM: Per capita crime rate by town
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town
CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX: Nitric oxide concentration (parts per 10 million)
RM: Average number of rooms per dwelling
AGE: Proportion of owner-occupied units built before 1940
DIS: Weighted distances to five Boston employment centers
RAD: Index of accessibility to radial highways
TAX: Full-value property tax rate per $10,000
PTRATIO: Pupil-teacher ratio by town
B: 1000(𝐵𝑘−0.63)^2  where Bk is the proportion of Black residents by town
LSTAT: Percentage of lower status of the population
MEDV: Median value of owner-occupied homes in $1000s (target variable)

## Installation
git clone https://github.com/yourusername/dragon-real-estate-price-predictor.git
cd dragon-real-estate-price-predictor
pip install -r requirements.txt

## Usage
## Load the Data:
python code
import pandas as pd
housing = pd.read_csv("housing_data.csv")

## Data Exploration: View the first few rows of the dataset:
housing.head()

## Model Training: Train and evaluate various regression models (e.g., Linear Regression, Decision Tree Regressor) to predict house prices. The code for training and evaluating models is included in the notebook.

## Results: Results are evaluated using RMSE (Root Mean Square Error) metrics. Insights and visualizations can be found in the Jupyter Notebook.

## Project Structure
housing_data.csv: Dataset used for training and evaluation.
notebook/Dragon_Real_Estate_Price_Predictor.ipynb: Jupyter notebook containing data preprocessing, model training, and evaluation.
requirements.txt: List of Python packages required for the project.

## Conclusion
This project demonstrates the application of regression models to predict house prices based on various real estate features. The model provides valuable insights for real estate investors and analysts.

## Future Work
Explore additional regression models or advanced techniques such as ensemble methods.
Incorporate more features or external datasets to improve prediction accuracy.
Deploy the model as a web application for real-time predictions.

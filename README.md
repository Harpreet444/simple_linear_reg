# Predicting Per Capita Income using Linear Regression

This project uses linear regression to predict the per capita income in Canada based on historical data. The dataset includes the year and corresponding per capita income values.

## Requirements

- pandas
- matplotlib
- numpy
- scikit-learn

## Dataset

The dataset used is `canada_per_capita_income.csv`, which contains information about the year and the per capita income in Canada.

## Steps

1. **Data Preparation**: Load the dataset and visualize the data using scatter plots.
2. **Model Training**: Train a linear regression model to predict per capita income.
3. **Prediction**: Use the trained model to predict per capita income for a given year.
4. **Visualization**: Visualize the data and the regression line using Matplotlib.

### Data Preparation

Load the dataset and visualize the data:

```python
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("canada_per_capita_income.csv")

# Display the first 5 rows of the dataset
print(df.head())

# Scatter plot of year vs. per capita income
plt.scatter(df['year'], df['per capita income (US$)'], color='g', marker='*')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.show()

# Price Prediction

A simple machine learning project using **linear regression** to predict diamond prices based on features like **carat** and **cut quality**.

## 📁 Dataset

Uses `diamond_dataset.csv` with features including:
- `carat`
- `cut` (Ideal, Premium, etc.)
- `price`

## 🧠 What It Does

- Computes **covariance** and **correlation** matrices
- Transforms data (`log(price)`, `sqrt(carat)`)
- Encodes `cut` as numeric
- Visualizes data with scatter plots and histograms
- Trains:
  - **Simple Linear Regression** (using `carat_root`)
  - **Multiple Linear Regression** (using `carat_root` + `cut_num`)
- Compares models with **R² scores**

## 📈 Output

- Color-coded scatter plot of carat vs. log price
- Histogram: predicted vs. actual prices
- R² values for model evaluation

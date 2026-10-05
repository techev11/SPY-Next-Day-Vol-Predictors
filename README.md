# S&P 500 Next-Day Volatility Prediction

A machine learning approach to forecasting next-day realized volatility of the **SPDR S&P 500 ETF (SPY)** using daily market data, macroeconomic indicators, technical indicators, and volatility features.

The goal of this project is to determine whether next-day volatility can be forecast effectively using **freely accessible daily data**, rather than relying on expensive high-frequency intraday datasets. The project also investigates whether nonlinear relationships among market, volatility, technical, and macroeconomic features provide meaningful predictive power for future volatility by comparing linear, tree-based, and neural network models.

## Overview

Accurate volatility forecasts are useful for portfolio construction, risk management, hedging, and derivatives trading.

This project constructs a feature-rich time-series dataset and compares four regression approaches:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Long Short-Term Memory (LSTM) neural network

The target is **next-day realized volatility of SPY**.

## Data

The dataset combines market and macroeconomic data from publicly accessible sources.

### Market Data

Daily market data includes:

* SPY OHLCV data
* CBOE Volatility Index (VIX)
* Short-term U.S. Treasury yields

### Macroeconomic Data

Macroeconomic features include:

* Federal Funds Effective Rate (FFER)
* Economic Policy Uncertainty Index

The primary dataset covers market observations from **1993–2025**.

## Target Variable

Daily log return is calculated as:

```text
r_t = log(P_t) - log(P_{t-1})
```

The project's daily realized-volatility proxy is the magnitude of the daily log return:

```text
RV_t = sqrt(r_t²) = |r_t|
```

The target is shifted one trading day forward so that information available at time `t` is used to predict:

```text
RV_{t+1}
```

## Feature Engineering

The model uses a broad feature set designed to capture volatility persistence, market conditions, momentum, intraday price behavior, and macroeconomic conditions.

Features include:

* Daily and lagged log returns
* Lagged realized volatility
* 5-day and 22-day rolling statistics
* SPY OHLCV data
* VIX OHLC data
* Open-to-close price changes
* Close-to-open overnight price changes
* High-low trading ranges
* Moving averages
* Momentum indicators
* Relative Strength Index (RSI)
* Bollinger Bands
* Rate of Change (ROC)
* Average True Range (ATR)
* Trading-volume features
* Federal Funds Effective Rate
* Economic Policy Uncertainty
* Short-term Treasury yields

## Train / Validation / Test Split

Because this is a time-series forecasting problem, the data is split **chronologically rather than randomly** to prevent look-ahead bias.

```text
Training:   70%
Validation: 15%
Test:       15%
```

The training window was designed to contain both normal and high-volatility market regimes, including the 2008 financial crisis.

## Models

### Linear Regression

Linear regression provides a simple baseline for determining whether more complex nonlinear models add predictive value.

### Random Forest

Random Forest is used to capture nonlinear relationships and interactions among the engineered market features while remaining relatively robust to noisy financial data.

### XGBoost

XGBoost uses gradient-boosted decision trees to capture nonlinear relationships while incorporating regularization to reduce overfitting.

The tuned model uses approximately:

```text
n_estimators = 1000
max_depth = 4
learning_rate = 0.07
subsample = 0.7
colsample_bytree = 0.6
reg_alpha = 0.5
reg_lambda = 2
min_child_weight = 5
```

### LSTM

A Long Short-Term Memory neural network is also evaluated to test whether a model explicitly designed for sequential data can outperform tree-based models using manually engineered time-series features.

The LSTM uses rolling **30-day sequences** as inputs.

## Results

| Model             |   Test RMSE |    Test MAE |     Test R² |
| ----------------- | ----------: | ----------: | ----------: |
| Linear Regression |     0.00835 |     0.00604 |    -0.16084 |
| Random Forest     |     0.00680 |     0.00498 |     0.23002 |
| XGBoost           | **0.00677** |     0.00483 | **0.23586** |
| LSTM              |     0.00688 | **0.00479** |      0.0817 |

XGBoost produced the strongest overall test performance, achieving an RMSE of **0.00677**, MAE of **0.00483**, and R² of **0.23586**.

The results suggest that nonlinear tree-based models can extract meaningful predictive information from daily market and macroeconomic features without requiring high-frequency intraday data.

## Feature Importance

VIX-derived variables were among the most important predictors for both Random Forest and XGBoost.

The five most important XGBoost features were:

1. `Low_VIX`
2. `High_VIX`
3. `Open_VIX`
4. `ATR`
5. `Close_VIX`

This confirms that options-implied volatility contains substantial information about subsequent realized market volatility.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* XGBoost
* PyTorch
* yfinance
* Matplotlib
* Jupyter Notebook


## Future Work

Potential extensions include:

* Real-time daily volatility forecasting
* Periodic or online model retraining
* Walk-forward validation
* Additional macroeconomic lags
* Improved LSTM architectures
* Attention-based or Transformer models
* Probabilistic volatility forecasts
* Testing across additional indices and asset classes





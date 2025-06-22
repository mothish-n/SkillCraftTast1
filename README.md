# House Price Prediction with Linear Regression

This project trains a simple Linear Regression model to predict house sale prices  
based on three features from the “House Prices: Advanced Regression Techniques” dataset:

- **GrLivArea**: Above-ground living area square footage  
- **BedroomAbvGr**: Number of bedrooms above ground  
- **FullBath**: Number of full bathrooms  

## Files

- `house_price_regression.py` – Python script that:
  1. Loads `train.csv` (downloaded from Kaggle)
  2. Trains a `LinearRegression` model
  3. Evaluates it (MSE & R²)
  4. Makes a sample prediction  
- `requirements.txt` – Python dependencies  
- `.gitignore` – files/folders to ignore  

## Setup & Run

1. **Clone** the repo:  
   ```bash
   git clone https://github.com/mothish-n/house-price-regression.git
   cd house-price-regression

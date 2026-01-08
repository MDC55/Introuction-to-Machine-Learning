# Introuction-to-Machine-Learning
## Boston House Price Prediction

Notebook: `Boston House Price Prediction.ipynb` — an end-to-end demo for exploring and modeling the (deprecated) Boston housing dataset.

## Summary
- Exploratory data analysis (pandas, seaborn, matplotlib), feature selection via correlation heatmap.
- Modeling: train/test split, LinearRegression, evaluation with RMSE.
- Common notebook patterns: global variables (X, y, data), plotting with `%matplotlib inline`, and use of log-transform for Sale_price.

## Key implementation details (examples from notebook)
- Dataset load (deprecated):
```python
from sklearn.datasets import load_boston
boston_dataset = load_boston()
X = boston_dataset.data
y = boston_dataset.target
```
- Dataframe creation:
```python
data = pd.DataFrame(X, columns=boston_dataset.feature_names)
data['Sale_price'] = y
```
- Train/test split and modeling:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression(); model.fit(X_train, y_train)
```

## Requirements (tested)
- Python 3.8+
- scikit-learn==1.1.3, pandas, numpy, seaborn, matplotlib, scipy, jupyter

Install (Windows):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install scikit-learn==1.1.3 pandas numpy seaborn matplotlib scipy jupyter
```

## Run
- Open in VS Code or:
```bash
jupyter notebook "Boston House Price Prediction.ipynb"
```
- Convert to script for CI / quick runs:
```bash
jupyter nbconvert --to script "Boston House Price Prediction.ipynb"
python "Boston House Price Prediction.py"
```

## Important note: `load_boston` is deprecated
- Options:
  - Pin scikit-learn to 1.1.x (used here).
  - Replace with `fetch_california_housing`:
    ```python
    from sklearn.datasets import fetch_california_housing
    X, y = fetch_california_housing(return_X_y=True)
    ```
  - Or use OpenML:
    ```python
    from sklearn.datasets import fetch_openml
    boston = fetch_openml(name="boston", as_frame=False)
    X, y = boston.data, boston.target
    ```

## Contributing notes
  - `fix: replace load_boston with fetch_california_housing`
  - `chore: add README and .gitignore`

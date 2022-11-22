import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import f1_score


# Loading dataset
df = pd.read_csv("/Users/arsenetripard/Downloads/dataproject2022.csv")
df.set_index("ID", inplace=True)

# Splitting into X and y
target = ["Pred_default (y_hat)"]
features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
X = df[features].to_numpy()
y = df[target].to_numpy().ravel()

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# HalvingGridSearchCV tp get best RandomForestClassifier as possible
param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
base_estimator = RandomForestClassifier(random_state=0)
search = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2, 
                         resource='n_estimators', max_resources=30).fit(X, y)
best_estimator = RandomForestClassifier(**search.best_params_)
best_estimator.fit(X_train, y_train)
print("="*80)
print(f"Best RandomForestClassifier uses these parameters {search.best_params_}\n")
print(f"mean accuracy score : {best_estimator.score(X_test, y_test):.2f}")
print(f"f1 score : {f1_score(y_test, best_estimator.predict(X_test)):.2f}")
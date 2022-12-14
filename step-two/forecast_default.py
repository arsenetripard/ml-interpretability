import pandas as pd
import numpy as np
import yaml 
import random
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve
from joblib import dump


# Reading DATA_FILEPATH from config.yml
with open(r'config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Loading dataset
df = pd.read_csv(config["DATA_FILEPATH"])
df.set_index("ID", inplace=True)

# Initialize Oversampler
sm = SMOTE(random_state=42)

# Splitting into X and y
target = ["Default (y)"]
features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
X = df[features].to_numpy()
y = df[target].to_numpy().ravel()

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Oversampling 
X_train, y_train = sm.fit_resample(X_train, y_train)

# HalvingGridSearchCV tp get best RandomForestClassifier as possible
param_grid = {'max_depth': [3, 5, 10, 12, 15], 'min_samples_split': [2, 4, 6, 8, 10]}
base_estimator = RandomForestClassifier(random_state=0)
search = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2, 
                         resource='n_estimators', max_resources=30).fit(X, y)
best_estimator = RandomForestClassifier(**search.best_params_)
best_estimator.fit(X_train, y_train)
print("="*80)
print(f"Best RandomForestClassifier uses these parameters {search.best_params_}\n")
print(f"mean accuracy score : {best_estimator.score(X_test, y_test):.2f}")
print(f"f1 score : {f1_score(y_test, best_estimator.predict(X_test)):.2f}")

# Saving the model
dump(best_estimator, 'best_estimator.joblib')

# Confusion Matrix 
print("="*80)
print(f"Confusion Matrix")
y_pred = best_estimator.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
matrix = ConfusionMatrixDisplay(confusion_matrix=cm) 
matrix.plot()

# ROC Curve
print("="*80)
print(f"ROC Curve with AUC score of {roc_auc_score(y_test , clf.predict_proba(X_test)[:, 1])}")
print(plot_roc_curve(best_estimator, X_test, y_test)) 


import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.inspection import PartialDependenceDisplay

features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
target = ["y_hat (own model)"]


# Reading DATA_FILEPATH from config.yml
with open(r'config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Loading dataset
df = pd.read_csv(config["DATA_FILEPATH"])
df.set_index("ID", inplace=True)

# Loading best_estimator
model = load("best_estimator.joblib")

# Adding the predictions proba from own model
df.rename(columns={"Pred_default (y_hat)": "y_hat (foreign model)"}, inplace=True)
df["y_hat (own model)"] = model.predict(df[features])

# Splitting into X and y
X = df[features] 
y = df[target] 

# Computing PartialDependanceDisplay
gridPlot = PartialDependenceDisplay.from_estimator(model, X, features=np.arange(stop=10),
    feature_names=features, kind='average')
plt.show()
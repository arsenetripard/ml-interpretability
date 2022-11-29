import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import load
from PyALE import ale

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

# Comute ALE
feature = "Age"
disp = ale(
    X=X[features], 
    model=model, 
    feature=[feature], 
    grid_size=50, 
    include_CI=True,
    C=0.95
)
plt.suptitle(f"ALE effect on feature : {feature}")
plt.gca().set_title("")
plt.show()
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression


# Reading DATA_FILEPATH from config.yml
with open(r'config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Loading dataset
df = pd.read_csv(config["DATA_FILEPATH"])
df.set_index("ID", inplace=True)

# Splitting into X and y
target = ["Pred_default (y_hat)"]
features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
X = df[features].to_numpy()
y = df[target].to_numpy()

# We will use LogisticRegression (LR) as surrogate
# Creating LR model and fitting it
clf = LogisticRegression()
clf.fit(X, y.ravel())
clf_coefs_ = clf.coef_.flatten()
model_coeffs = {"features": features, "coefficients": np.round(clf_coefs_, 2)}
interpretability = pd.DataFrame(data=model_coeffs)
print("="*80)
print("Model coefficients:\n")
print(interpretability)

# Taking a random individual and printing each feature's contribution
print("="*80)
individual = df[features].sample(1).to_numpy()
print(f"One individual is chosen at random. Our model predicts a default probability of {clf.predict_proba(individual)[0][1]:.2f}\n")
individual = individual.ravel()
features_contributions = individual * clf_coefs_
interpretability["individual"] = np.round(individual, 2)
interpretability["features_contributions"] = np.round(features_contributions, 2)

print("Features contributions:\n")
print(interpretability)
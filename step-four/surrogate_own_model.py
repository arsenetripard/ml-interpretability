import numpy as np
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from joblib import load

features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]


# Reading DATA_FILEPATH from config.yml
with open(r'config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Loading dataset
df = pd.read_csv(config["DATA_FILEPATH"])
df.set_index("ID", inplace=True)

# Loading best_estimator
model = load("best_estimator.joblib")

# Adding the predictions of our model
df["y_hat (foreign model)"] = df["Pred_default (y_hat)"]
df.drop(columns="Pred_default (y_hat)", inplace=True)
df["y_hat (own model)"] = model.predict(df[features])
print("="*80)
print(df["y_hat (own model)"].sample(10, random_state=10), "\n")
print(df["y_hat (foreign model)"].sample(10, random_state=10), "\n")

# Splitting into X and y
target = ["y_hat (own model)"]
X = df[features].to_numpy()
y = df[target].to_numpy()

# Surrogate OWN model
own_clf = LogisticRegression()
own_clf.fit(X, y.ravel())
own_clf_coefs_ = own_clf.coef_.flatten()
own_model_coeffs = {"features": features, "own_model_coefficients": np.round(own_clf_coefs_, 2)}
own_model_interpretability = pd.DataFrame(data=own_model_coeffs)

# Surrogate FOREIGN model
foreign_clf = load("surrogate_foreign_model.joblib")
foreign_clf_coefs_ = foreign_clf.coef_.flatten()
foreign_model_coeffs = {"features": features, "foreign_model_coefficients": np.round(foreign_clf_coefs_, 2)}
foreign_model_interpretability = pd.DataFrame(data=foreign_model_coeffs)

# Compare surrogates
individual = df[features].sample(1).to_numpy().ravel()
own_model_ft_contributions = individual * own_clf_coefs_
foreign_model_ft_contributions = individual * foreign_clf_coefs_
interpretability = own_model_interpretability.merge(foreign_model_interpretability, how="left", on="features")
interpretability["individual"] = np.round(individual, 2)
interpretability["own_model_ft_contributions"] = np.round(own_model_ft_contributions, 2)
interpretability["foreign_model_ft_contributions"] = np.round(foreign_model_ft_contributions, 2)
print("="*80)
print("Compare features contributions between foreign and own models:\n")
print(interpretability[["features", "individual", "own_model_coefficients", 
    "own_model_ft_contributions", "foreign_model_coefficients", "foreign_model_ft_contributions"]], "\n")

interpretability = pd.melt(interpretability[["features", "own_model_ft_contributions", 
    "foreign_model_ft_contributions"]], id_vars=["features"], 
    value_vars=["own_model_ft_contributions", "foreign_model_ft_contributions"])
sns.catplot(data=interpretability, kind="bar", x="features", y="value", hue="variable")
plt.xticks(rotation=30)
plt.show()
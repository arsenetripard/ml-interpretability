import numpy as np
import pandas as pd
import yaml
import os
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE 
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
from sklearn.inspection import plot_partial_dependence, PartialDependenceDisplay
from PyALE import ale
from scipy.stats import chi2_contingency, chi2
from aif360.sklearn import metrics
import lime
import lime.lime_tabular
import shap


# Reading DATA_FILEPATH from config.yml
os.chdir('..')
with open(r'config.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Loading dataset
df = pd.read_csv(config["DATA_FILEPATH"])
df.set_index("ID", inplace=True)

data = df

#####################################################################################################################
##############################     STEP 1    ########################################################################
#####################################################################################################################

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

# Saving the surrogate model for Step 4
print("="*80)
print("Saving the surrogate model...")
dump(clf, "surrogate_foreign_model.joblib")
print("Saved!")

#####################################################################################################################
##############################     STEP 2    ########################################################################
#####################################################################################################################


# Initialize Oversampler
sm = SMOTE(random_state=42)

# Splitting into X and y
target = ["Default (y)"]
features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
X = data[features].to_numpy()
y = data[target].to_numpy().ravel()

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


#####################################################################################################################
##############################     STEP 4    ########################################################################
#####################################################################################################################

df = data

# Load Model
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


#####################################################################################################################
##############################     STEP 5    ########################################################################
#####################################################################################################################

features = ["Job tenure", "Age", "Car price", "Funding amount", "Down payment",
            "Loan duration", "Monthly payment", "Credit event", "Married", "Homeowner"]
target = ["y_hat (own model)"]

df = data 

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


#####################################################################################################################
##############################     STEP 6    ########################################################################
#####################################################################################################################

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


#####################################################################################################################
##############################     STEP 7    ########################################################################
#####################################################################################################################

for i in range(10):
    print(plot_partial_dependence(best_estimator, X_test, features = [i] ,feature_names = features, kind='individual', target = "PD"))

#####################################################################################################################
##############################     STEP 8    ########################################################################
#####################################################################################################################

explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                    feature_names=features, 
                    class_names=['PD'], 
                    # categorical_features=, 
                    # There is no categorical features in this example, otherwise specify them.                               
                    verbose=True, mode='regression')

exp = explainer.explain_instance(X_test[0], 
     best_estimator.predict, num_features=len(features))
print(exp.as_pyplot_figure())

#####################################################################################################################
##############################     STEP 9    ########################################################################
#####################################################################################################################

explainer = shap.Explainer(best_estimator)
shap_test = explainer(X_test)
print(f"Shap values length: {len(shap_test)}\n")
print(f"Sample shap value:\n{shap_test[0]}")
print(f"Expected value: {explainer.expected_value[1]:.2f}")
print(f"Average target value (training data): {y_train.values.mean():.2f}")
print(f"Base value: {np.unique(shap_test.base_values)[0]:.2f}")

shap_df = pd.DataFrame(shap_test.values[:,:,1], 
                       columns=features)

# Checking if the sum of the shap values (+ the base expected value) equal the prediction"
np.isclose(best_estimator.predict_proba(X_test)[:,1], 
           explainer.expected_value[1] + shap_df.sum(axis=1))

# # Shap 1 - Global - Bar plot
print(shap.plots.bar(shap_test[:,:,1]))

# Similar analysis than before, Job tenure shows great importance, same for homeowner etc.
# # Shap 2 - Global - Summary plot
print(shap.summary_plot(shap_test[:,:,1]))

# # Shap 3 - Global - Heatmap
print(shap.plots.heatmap(shap_test[:,:,1]))

# # Shap 4 - Global - Force plot
shap.initjs()
print(shap.force_plot(explainer.expected_value[1], shap_test[:,:,1].values, 
                X_test))

#####################################################################################################################
##############################     STEP 10    #######################################################################
#####################################################################################################################

# Preparing Data
df = pd.DataFrame(X_test, columns=features).join(data['Group']).join(data['Default (y)'])
df['privileged'] = np.where(df['Age'] < 50, 1, 0)

prediction = pd.DataFrame(y_pred, columns=['prediction']).set_index(df.index)
df = df.join(prediction)

# Statistical Parity
print(f"Value of Statistical Parity Difference : {metrics.statistical_parity_difference( y_true=y_test, y_pred=df['prediction'], prot_attr=df["privileged"], priv_group=1, pos_label=1)}")
# Chi-square test of independence.
contigency= pd.crosstab(df['privileged'], df['prediction']) 
print(f"CrossTab contingency : {contingency}") 
c, p, dof, expected = chi2_contingency(contigency) 
print(f"We have a P-value for the Statistical Parity of {p:.2f}")                                                                                           


# Conditional Parity
print(f"Value of Conditional Parity Difference : {metrics.conditional_demographic_disparity(  y_true=y_test, y_pred=df['prediction'], prot_attr=df["privileged"], pos_label=1)}")

# Chi-square test of independence.
contigency_0 = pd.crosstab(df.loc[df['Group'] == 0, 'privileged'], df.loc[df['Group'] == 0, 'prediction'])
contigency_1 = pd.crosstab(df.loc[df['Group'] == 1, 'privileged'], df.loc[df['Group'] == 1, 'prediction'])

c0, p, dof, expected = chi2_contingency(contigency_0) 
c1, p, dof, expected = chi2_contingency(contigency_1)

print(f"We have a P-value for the Conditional Parity Difference of {chi2.sf(c1 + c0, 2)}")


# Equalized Odds 
print(f"Value of Average Odds Difference : {metrics.average_odds_difference( y_true=df['Default (y)'], y_pred=df['prediction'], prot_attr=df['privileged'], priv_group=1, pos_label=1)}")

#####################################################################################################################
##############################     STEP 11    #######################################################################
#####################################################################################################################

# Home Owner
stat = []
home_owner = [0, 1]
for i in home_owner: 
    temp = df[features].copy()
    temp['Homeowner'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

bar = sns.barplot(pd.DataFrame([stat], columns=['Equals 0', 'Equals 1']))
print(f"Home Owner FPDP {bar.set(xlabel ="HomeOwner", ylabel = "Average Odds Difference")}")


# Credit Event
stat = []
credit_event = [0, 1]
for i in credit_event: 
    temp = df[features].copy()
    temp['Credit event'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

bar = sns.barplot(pd.DataFrame([stat], columns=['Equals 0', 'Equals 1']))
print(f"Credit Event FPDP {bar.set(xlabel ="Credit Event", ylabel = "Average Odds Difference")}")


# Down Payment
stat = []
down_payment = [0, 1]
for i in down_payment: 
    temp = df[features].copy()
    temp['Down payment'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

bar = sns.barplot(pd.DataFrame([stat], columns=['Equals 0', 'Equals 1']))
print(f"Down Payment FPDP {bar.set(xlabel ="Down Payment", ylabel = "Average Odds Difference")}")


# Marriage
stat = []
married = [0, 1]
for i in married:
    temp = df[features].copy()
    temp['Married'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)


bar = sns.barplot(pd.DataFrame([stat], columns=['Unmarried', 'Married']))
print(f"Marriage FPDP {bar.set(xlabel ="Married", ylabel = "Average Odds Difference")}")


# Job Tenure
stat = []
job = range(min(df['Job tenure']), max(df['Job tenure']))
for i in job: 
    temp = df[features].copy()
    temp['Job tenure'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = job
graph = sns.lineplot(stat)
print(f"Job Tenure FPDP {graph.set(xlabel ="Job tenure", ylabel = "Average Odds Difference")}")


# Loan Duration
stat = []
loan_duration = range(min(df['Loan duration']), max(df['Loan duration']))
for i in loan_duration: 
    temp = df[features].copy()
    temp['Loan duration'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = loan_duration
graph = sns.lineplot(stat)
print(f"Loan Duration FPDP {graph.set(xlabel ="Loan Duration", ylabel = "Average Odds Difference")}")


# Age
stat = []
age = range(min(df['Age']), max(df['Age']))
for i in age: 
    temp = df[features].copy()
    temp['Age'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = age
graph = sns.lineplot(stat)
print(f"Age FPDP {graph.set(xlabel ="Age", ylabel = "Average Odds Difference")}")


# Monthly Payment
stat = []
monthly_payment = np.linspace(min(df['Monthly payment']), max(df['Monthly payment']), 100)
for i in monthly_payment: 
    temp = df[features].copy()
    temp['Monthly payment'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = monthly_payment
graph = sns.lineplot(stat)
print(f"Monthly Payment FPDP {graph.set(xlabel ="Monthly Payment", ylabel = "Average Odds Difference")}")


# Funding Amount
stat = []
funding_amount = np.linspace(min(df['Funding amount']), max(df['Funding amount']), 100)
for i in funding_amount: 
    temp = df[features].copy()
    temp['Funding amount'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = funding_amount
graph = sns.lineplot(stat)
print(f"Funding Amount FPDP {graph.set(xlabel ="Funding amount", ylabel = "Average Odds Difference")}")

# Car Price
stat = []
car_price = np.linspace(min(df['Car price']), max(df['Car price']), 100)
for i in car_price: 
    temp = df[features].copy()
    temp['Car price'] = i
    prediction = pd.DataFrame(best_estimator.predict(temp), columns=['prediction']).set_index(temp.index)
    temp = temp.join(prediction)
    temp['privileged'] = df['privileged']
    temp['true'] = df['Default (y)']
    m = metrics.average_odds_difference(    y_true=temp['true'], 
                                            y_pred=temp['prediction'], 
                                            prot_attr=temp['privileged'], 
                                            priv_group=1, 
                                            pos_label=1)   
    stat.append(m)

stat = pd.Series(stat)
stat.index = car_price
graph = sns.lineplot(stat)
print(f"Car Price FPDP {graph.set(xlabel ="Car Price", ylabel = "Average Odds Difference")}")
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

df_h = pd.read_csv('heart.csv')

col_list = list(df_h.columns)
cat_vars = ['fbs', 'sex', 'cp', 'exang', 'restecg', 'slope', 'ca', 'thal']
num_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


X = df_h.copy()
null = X.isnull().sum()
y = X.pop('target')

plt.figure(figsize=(10,8))
sns.heatmap(X.astype(float).corr(), linewidths=0.1,
           square=True,  linecolor='white', annot=True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
#cat_4_treeModels = OneHotEncoder(handle_unknown='ignore')

#cat_prepro = ColumnTransformer(transformers=[('cat', cat_4_treeModels, cat_vars)])
model = CatBoostClassifier(learning_rate=0.1)

#pipe = make_pipeline([cat_prepro,model])
model.fit(X_train, y_train)
model.score(X_test, y_test)

r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
#df_r = pd.DataFrame(r, index=None)

for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{X.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")


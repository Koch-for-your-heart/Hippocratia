import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import category_encoders as ce
from sklearn import set_config
import pickle

set_config(display='diagram')

df = pd.read_csv('heart.csv')

X = df.copy()
y = X.pop('target')

cat_vars = ['fbs', 'sex', 'cp', 'exang', 'restecg', 'slope', 'ca', 'thal']
num_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

numpipe1 = Pipeline([('quantTrans', QuantileTransformer(random_state=7, n_quantiles=10, output_distribution='normal'))])

numpipe2 = Pipeline([('stdSle', StandardScaler())])

num_pipes = [numpipe1, numpipe2]


cat_enc = {'catBoost': ce.CatBoostEncoder(),
           'count': ce.CountEncoder(),
           'ordinal': ce.OrdinalEncoder(),
           'target': ce.TargetEncoder()}

def make_tree_prepro(x, y, z):
    pipe_cat = Pipeline([(f'{z}', y)])
    return ColumnTransformer(transformers=[('num', x, num_vars),
                                           (f'{z}', pipe_cat, cat_vars)],
                             remainder='drop')


l_prepro = []
for pipe in num_pipes:
    for cat, enc in enumerate(cat_enc):
        l_prepro.append(make_tree_prepro(pipe, cat_enc[enc], enc))

models = {'GradientBC': GradientBoostingClassifier(),
          'RandFC': RandomForestClassifier(),
          'SVC': SVC(),
          'CatBC': CatBoostClassifier(),
          'AdaBC': AdaBoostClassifier()}


model_pipe = make_pipeline(l_prepro[0], models['GradientBC'])

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

l_pipes = []
for prepro in l_prepro:
    for _, model in enumerate(models):
        l_pipes.append(Pipeline(steps=[('preprocess', prepro), ('model_name', models[model])]))


results = pd.DataFrame({'MSE': [], 'RMSE': []})
# for pipe in l_pipes:
#     #pipe.get_params()
#     pipe.fit(X_train, y_train)
#     preds = pipe.predict(X_test)
#
#     mse = sum((y_test - preds) ** 2) / preds.shape[0]
#     rmse = np.sqrt(mse)
#     results = results.append({"MSE": mse,
#                               "RMSE": rmse}, ignore_index=True)

best_model = l_pipes[3]

cat_boost_model = CatBoostClassifier(random_state=7,class_weights={0:1,1:3})
best_tras = ColumnTransformer(transformers=[('num', numpipe1, num_vars),
                                           ('cat', cat_enc['catBoost'], cat_vars)],
                                            remainder='drop')
best_pipe = make_pipeline(best_tras,cat_boost_model)
param_list = best_pipe.get_params()
print(best_pipe.get_params())
params = {'learning_rate':[0.1,0.001],
          }
grid = GridSearchCV(estimator=best_pipe,
                    cv=skf,
                    refit=True,
                    n_jobs=-1,
                    verbose=3,
                    param_grid={})

grid.fit(X_train, y_train)
cv_grid = pd.DataFrame(grid.cv_results_)
pred = grid.predict(X_test)

acc = accuracy_score(y_test, pred)
matrix = confusion_matrix(y_test, pred)

new_data = pd.read_csv('only_augumented_heart_data.csv')

pred2 = grid.predict(new_data)
new_preds = pd.Series(pred2, index=None)
new_data['target'] = new_preds
list_of_df = [df, new_data]
aug_data = pd.concat(list_of_df)
X_aug = aug_data.copy()
y_aug = X_aug.pop('target')

X_aug_train, X_aug_val, y_aug_train, y_aug_val = train_test_split(X_aug,y_aug,test_size=0.2,stratify=y_aug,random_state=4)

grid.fit(X_aug_train,y_aug_train)
last_preds = grid.predict(X_aug_val)
last_acc = accuracy_score(y_aug_val,last_preds)
last_matrix = confusion_matrix(y_aug_val,last_preds)
grid.fit(X_aug,y_aug)
filename = 'model.sav'
pickle.dump(grid,open(filename,'wb'))

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt

#Load in csv file into pandas
train_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv')
test_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\test.csv')

#Clean Data/(X_train, y_train, X_valid, y_valid, my_cols)
Names = train_data['Name'].to_numpy()
X_train_full = train_data.drop(['Survived','Cabin','Name'],axis=1)
y_train = train_data.Survived

X_test = test_data.drop(['Name','Cabin'],axis=1)

#Organize data into categorical_cols+numerical_cols
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer.fit(X_train[numerical_cols])
categorical_transformer.fit(X_train[categorical_cols])
numerical_transformer.transform(X_train[numerical_cols])
categorical_transformer.transform(X_train[categorical_cols])

model = xgb.XGBClassifier(max_depth=5, n_estimators = 500, learning_rate = 0.01, random_state = 0)

model.fit(X_train, y_train)

print(type(model))
survivors = pd.dataframe(y_train, columns = ['Survived'])

final_df = pd.concat([X_train, survivors])

plt.scatter(x=X_train['Pclass'], y = X_train['Age'], label = Names)
plt.title('Scatter plot of Pclass and Sex')
plt.xlabel('Pclass')
plt.ylabel('Sex')
plt.show()
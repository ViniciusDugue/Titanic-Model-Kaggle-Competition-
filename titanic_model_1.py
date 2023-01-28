import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

#Load in csv file into pandas
titanic_train_file_path= r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv'
train_data = pd.read_csv(titanic_train_file_path)

# Clean Data/ split predictors and target variable /(X_train, y_train, X_valid, y_valid, my_cols)
X = train_data.drop(['Survived','Name','Cabin'],axis=1)
y = train_data.Survived
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#Organize data into categorical_cols+numerical_cols
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

#Preprocess Data
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

#Create Model
model = RandomForestRegressor(n_estimators=50, random_state=0)

#Preprocess data with pipeline 
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)#MAE
print('MAE:', score)
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')#Cross-Validation
print("MAE scores:\n", scores)
print(f"Average MAE score (across experiments): {scores.mean()}")

#Display Data info
# pd.options.display.max_columns = None
# print(X_train_full.head(8))
# print(train_data.describe())



'''
gradientboosting classifier,pipeline, xgb_parameter_grid, cross-validation, matplotlib'''

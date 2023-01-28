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

#Load in csv file into pandas
train_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv')
test_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\test.csv')

#Clean Data/(X_train, y_train, X_valid, y_valid, my_cols)
X_train_full = train_data.drop(['Survived','Name','Cabin'],axis=1)
y_train = train_data.Survived
X_test = test_data.drop(['Name','Cabin'],axis=1)

#Organize data into categorical_cols+numerical_cols
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()

#Create Transformers for data processing
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

#Create Model
model = xgb.XGBClassifier()

#Preprocess data with pipeline 
my_pipeline =Pipeline(steps= [('preprocessor', preprocessor),
    ('standard_scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('model', model)])

#Create Param_grid to find best Model
param_grid = {
    'pca__n_components':[5, 7, 10, 11] ,
    'model__max_depth': [2, 3, 5, 7, 10],
    'model__n_estimators': [10, 100, 500],
    'model__learning_rate':[0.01,0.03,0.05,0.1]
}

#Create GridSearchCV and fit to find best predictions
grid = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid.fit(X_train, y_train)
grid2 = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
grid2.fit(X_train, y_train)

# Evaluate the model 
mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]# mean of cross validation scores using roc_auc scoring metric
std_score = grid.cv_results_["std_test_score"][grid.best_index_]
print("Mean_score(roc_auc):\n", mean_score)
print("Std_score(roc_auc):\n", std_score)
mean_score2 = grid2.cv_results_["mean_test_score"][grid2.best_index_]# mean of cross validation scores using roc_auc scoring metric
std_score2 = grid2.cv_results_["std_test_score"][grid2.best_index_]
print("MAE:\n", -1*mean_score2)
print("Std_score(MAE):\n", std_score2)

#Predict from Test Data
test_preds = grid2.predict(X_test) 

#Submission
submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':test_preds})
filename = 'Titanic Predictions 1.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

#Display data info
#pd.options.display.max_columns = None
#print(X_train_full.head(8))
#print(train_data.describe())

#count_passenger=len(pd.read_csv)  group survival rate not useful because we are predicting individual survival rate

'''
gradientboosting classifier,pipeline, xgb_parameter_grid, cross-validation, matplotlib'''
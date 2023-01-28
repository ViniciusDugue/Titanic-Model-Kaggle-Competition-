import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load in csv file into pandas
train_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv')
test_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\test.csv')

#Clean Data/(X_train, y_train, X_valid, y_valid, my_cols)
X_train_full = train_data.drop(['Survived','Name','Cabin'],axis=1)
y= train_data.Survived

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

model = xgb.XGBClassifier()
#Preprocess data with pipeline 
X = preprocessor.fit_transform(X_train_full)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

my_pipeline =Pipeline(steps= [
    ('standard_scaler', StandardScaler()), 
    ('pca', PCA()), 
    ('model', model)]) 

#maybe add early_stopping_rounds to grid, and learning_rate to model??
param_grid = {
    'pca__n_components':[5, 7, 10, 11] ,#should maybe set it to auto??
    'model__max_depth': [2, 3, 5, 7, 10],
    'model__n_estimators': [10, 100, 500],
    'model__learning_rate':[0.01,0.03,0.05,0.1],
    'model__early_stopping_rounds': [3,5,10,15]
}
######MAKE SURE TO PREDICT???#######  ,'model__early_stopping_rounds': [5, 10, 15] //, model__eval_set=[(X_valid, y_valid)]

grid = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

grid.fit(X_train, y_train,model__eval_set=[(X_valid,y_valid)])

grid2 = GridSearchCV(my_pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
grid2.fit(X_train, y_train,model__eval_set=[(X_valid,y_valid)])

# Evaluate the model 
mean_score = grid.cv_results_["mean_test_score"][grid.best_index_]# mean of cross validation scores using roc_auc scoring metric
std_score = grid.cv_results_["std_test_score"][grid.best_index_]
print("Mean_score(roc_auc):\n", mean_score)
print("Std_score(roc_auc):\n", std_score)

mean_score2 = grid2.cv_results_["mean_test_score"][grid2.best_index_]# mean of cross validation scores using roc_auc scoring metric
std_score2 = grid2.cv_results_["std_test_score"][grid2.best_index_]
print("MAE:\n", -1*mean_score2)
print("Std_score(MAE):\n", std_score2)

#Display data info
#pd.options.display.max_columns = None
#print(X_train_full.head(8))
#print(train_data.describe())

#count_passenger=len(pd.read_csv)  group survival rate not useful because we are predicting individual survival rate

'''
gradientboosting classifier,pipeline, xgb_parameter_grid, cross-validation, matplotlib'''
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
import numpy as np
import seaborn as sns


pd.options.display.max_columns = None
#PC files
# train_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv')
# test_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\test.csv')
#Laptop files
train_data = pd.read_csv(r'C:\Users\vinic\OneDrive\Desktop\Titanic Competition(Kaggle)\train.csv')
test_data = pd.read_csv(r'C:\Users\vinic\OneDrive\Desktop\Titanic Competition(Kaggle)\test.csv')


Full_data = pd.concat([train_data, test_data])

#Impute Missing Age by grouping by Sex, Pclass and using the median of the group
age_by_pclass_sex = Full_data.groupby(['Sex', 'Pclass']).median()['Age']
for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print(f'Median age of Pclass {pclass} {sex}s: {age_by_pclass_sex[sex][pclass]}')
print(f"Median age of all passengers: {Full_data['Age'].median()}")

#Impute 2 Missing Embarked
Full_data['Embarked'] = Full_data['Embarked'].fillna('S')

#Impute 1 Missing Fare by grouping by pclass,parch, sibSp and using the median of the group
med_fare = Full_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
Full_data['Fare']=Full_data['Fare'].fillna(med_fare)


#Creating Feature(Deck) by taking first letter of Cabin and if null imputing with 'M'
Full_data['Deck'] = Full_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

# One passenger in Deck T is 1st class so change to Deck A for ease
idx = Full_data[Full_data['Deck'] == 'T'].index
Full_data.loc[idx, 'Deck'] = 'A'

#Create bar graph displaying survival counts for each deck
Bar_data = Full_data.dropna(subset=['Survived'])
Bar_data['Survived'] = Bar_data['Survived'].astype(int)

print(Bar_data['Survived'])
deck_survived = {'A':[0,0],'B':[0,0],'C':[0,0],'D':[0,0],'E':[0,0],'F':[0,0],'G':[0,0]}

for index, row in Bar_data.iterrows():
    deck = row['Deck']
    survived = row['Survived']
    if deck in deck_survived:
        deck_survived[deck][survived]+=1

decks = list(deck_survived.keys())
survived_count = [deck_survived[deck][1] for deck in decks]
not_survived_count = [deck_survived[deck][0] for deck in decks]

fig,ax = plt.subplot()
ax.bar(decks, survived_count, label='Survived')
ax.bar(decks, not_survived_count, bottom=survived_count, label='Not Survived')

ax.set_ytitle('Survival Count')
ax.set_xtitle('Deck')
ax.set_title('Survival Counts for Each Deck')





# df_all_decks_survived = Full_data.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
# 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name':'Count'}).transpose()
# print(['Deck', 'Survived'].count())

# X_train_full = Full_data.iloc[0:len(train_data)].drop(['Survived'], axis=1)
# X_train = train_data.drop(['Survived'],axis=1)
# y_train = train_data.Survived
# X_test = Full_data.iloc[len(train_data):len(Full_data)]

# #Organize data into categorical_cols+numerical_cols
# categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
# numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# my_cols = categorical_cols + numerical_cols
# X_train = X_train_full[my_cols].copy()

# numerical_transformer = SimpleImputer(strategy='mean')
# categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# numerical_transformer.fit(X_train[numerical_cols])
# categorical_transformer.fit(X_train[categorical_cols])
# numerical_transformer.transform(X_train[numerical_cols])
# categorical_transformer.transform(X_train[categorical_cols])

# model = xgb.XGBClassifier(max_depth=5, n_estimators = 500, learning_rate = 0.01, random_state = 0)

# model.fit(X_train, y_train)

plt.show()
#test edit 3
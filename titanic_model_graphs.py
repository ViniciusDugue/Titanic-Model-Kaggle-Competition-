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

import matplotlib.pyplot
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
#PC files
train_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\train.csv')
test_data = pd.read_csv(r'C:\Users\Vinicius Dugue\Desktop\Titanic Competition(Kaggle)\test.csv')
#Laptop files
# train_data = pd.read_csv(r'C:\Users\vinic\OneDrive\Desktop\Titanic Competition(Kaggle)\train.csv')
# test_data = pd.read_csv(r'C:\Users\vinic\OneDrive\Desktop\Titanic Competition(Kaggle)\test.csv')

print(train_data.head())
#Clean Data/(X_train, y_train, X_valid, y_valid, my_cols)

train_data['Alone']=0
train_data.loc[(train_data['SibSp']==0) & (train_data['Parch']==0) , 'Alone'] = 1

pd.options.display.max_columns = None
print(train_data.head(5))
Names = train_data['Name'].to_numpy()
X_train_full = train_data.drop(['Survived','Cabin','Name'],axis=1)
y_train = train_data.Survived

X_test = test_data.drop(['Name','Cabin'],axis=1)
Full_data = pd.concat([train_data, test_data]).drop(['Survived', 'Alone'], axis=1)

#Organize data into categorical_cols+numerical_cols
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()

survivors = pd.DataFrame(y_train, columns = ['Survived'])

final_df = pd.concat([X_train, survivors])

#Correlation Matrix with Heatmap
corr = Full_data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

#Calculating Values for Piecharts
num_children = len(train_data[train_data['Age'] < 18])
num_men = len(train_data[(train_data['Sex'] == 'male') & (train_data['Age'] > 17)])
num_women = len(train_data[(train_data['Sex'] == 'female') & (train_data['Age'] > 17)])
x1 = [num_men, num_children, num_women]
num_children_survived = train_data.query("Survived == 1 & Age < 18").shape[0]
num_men_survived = train_data.query("Survived == 1 & Sex == 'male' & Age >= 18").shape[0]
num_women_survived = train_data.query("Survived == 1 & Sex == 'female' & Age >= 18").shape[0]
x2 = [num_men_survived, num_children_survived, num_women_survived]
num_children_died = train_data.query("Survived == 0 & Age < 18").shape[0]
num_men_died = train_data.query("Survived == 0 & Sex == 'male' & Age >= 18").shape[0]
num_women_died = train_data.query("Survived == 0 & Sex == 'female' & Age >= 18").shape[0]
x3 = [num_men_died, num_children_died, num_women_died]

#Creating figure/ for loop to plot all piecharts
fig, axs = plt.subplots(3, 3, figsize=(7,12))
fig.subplots_adjust(wspace=1)
labels = ['Men', 'Children', 'Women']
x=[x1,x2,x3]
titlez = ['# of Total Men/Women/Children', '# of Men/Women/Children who Survived', '# of Men/Women/Children who died' ]
for i in range(len(x)):
    axs[i][0].pie(x[i], labels=labels, autopct='%1.1f%%', radius = 1.6, textprops={'fontsize': 7} )
    axs[i][0].set_title(titlez[i],fontsize=7, loc ='center')

#Creating Alone/Not Alone Piechart
num_alone = train_data.query("Alone == 1").shape[0]
num_not_alone = train_data.query("Alone == 0").shape[0]
x4 = [num_alone, num_not_alone]
axs[0][1].pie(x4, labels = ['Alone', 'Not Alone'], colors = sns.color_palette('pastel')[0:5], autopct='%1.1f%%', radius = 1.7, textprops={'fontsize': 14})

#Creating bar plot for all missing values of data
plt.subplots_adjust(wspace=1.5)
msno.matrix(Full_data, figsize=[8, 8], fontsize=10)

plt.show()

# plt.scatter(x=X_train['Pclass'], y = X_train['Age'], label = Names)
# plt.title('Scatter plot of Pclass and Sex')
# plt.xlabel('Pclass')
# plt.ylabel('Sex')
# plt.show()

#first edit
#second edit

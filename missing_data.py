import os

from sklearn.preprocessing import OrdinalEncoder

path = "/Users/lj/ML_Python/HousePrice"
for dirname, _, filenames in os.walk(path+'/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression,Lasso,Ridge,BayesianRidge
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")

# Read files
df_train = pd.read_csv(path+"/data/train.csv")
df_test = pd.read_csv(path+"/data/test.csv")
submission = pd.read_csv(path+"/data/sample_submission.csv")
df = pd.concat([df_train,df_test],axis = 0,ignore_index = True)
df["LogSalePrice"] = np.log10(df["SalePrice"])
## Lets create a list of column names for categorical and numerical features

cate_feat = list(df.select_dtypes(include=[object]).columns)
num_feat = list(df.select_dtypes(include=[int,float]).columns)

## Handling Missing Data
## Number of missing values in categorical features
print(df[cate_feat].isnull().sum())

## Number of missing values in numerical features
print(df[num_feat].isnull().sum())

# Dealing with null values in numerical features
# Since there is alot missing in lotfrontage, and is related to lotArea as seen below, we will use linear reg to fill in the missing values for LotFrontage
sns.lmplot(x="LotArea",y="LotFrontage",data = df)
plt.ylabel("LotFrontage")
plt.xlabel("LotArea")
plt.title("LotArea vs LotFrontage")
plt.show()

lm = LinearRegression()
lm_X = df[df['LotFrontage'].notnull()]['LotArea'].values.reshape(-1,1)
lm_y = df[df['LotFrontage'].notnull()]['LotFrontage'].values
lm.fit(lm_X,lm_y)
df['LotFrontage'].fillna((df['LotArea'] * lm.coef_[0] + lm.intercept_), inplace=True)
df['LotFrontage'] = df['LotFrontage'].apply(lambda x: int(x))

#Since its only few values missing in each category, we dont need to explore alot. Just filling na values with median and mean
df["GarageYrBlt"].fillna(df["GarageYrBlt"].median(),inplace = True)
df["BsmtFinSF1"].fillna(df["BsmtFinSF1"].mean(),inplace = True)
df["BsmtFinSF2"].fillna(df["BsmtFinSF2"].mean(),inplace = True)
df["BsmtUnfSF"].fillna(df["BsmtUnfSF"].mean(),inplace = True)
df["TotalBsmtSF"].fillna(df["TotalBsmtSF"].mean(),inplace = True)
df["BsmtFullBath"].fillna(df["BsmtFullBath"].median(),inplace = True)
df["BsmtHalfBath"].fillna(df["BsmtHalfBath"].median(),inplace = True)
df["GarageArea"].fillna(df["GarageArea"].mean(),inplace = True)
df["GarageCars"].fillna(int(df["GarageCars"].median()),inplace = True)
df["MasVnrArea"].fillna(df["MasVnrArea"].median(),inplace = True)

# Dealing with null values in Categorical features
## Since more than 90% of the column is null
df.drop(["Alley","FireplaceQu","PoolQC","MiscFeature"], axis = 1,inplace = True)
df.drop(["Fence"], axis = 1,inplace = True)

# Remove the items from the column list
for item in cate_feat:
    if item in ["Alley","FireplaceQu","PoolQC","MiscFeature"]:
        cate_feat.remove(item)
cate_feat.remove("Fence")

# Fill null values with none for items that are missing the item and mode for the rest of the missing values
cate_none = ["BsmtExposure", "BsmtFinType2", "BsmtCond", "BsmtQual", "BsmtFinType1"]
cate_mode = ["Electrical", "Functional", "KitchenQual", "Exterior1st", "Exterior2nd", "MSZoning", "SaleType", "MasVnrType", "GarageFinish", "GarageQual", "GarageCond", "GarageType","Utilities"]

for col in cate_none:
    df[col].fillna('none',inplace = True)

for col in cate_mode:
    df[col].fillna(df[col].mode()[0],inplace = True)

print(f"Null values: {df.drop(['SalePrice','LogSalePrice'],axis = 1).isnull().sum().sum()}")

'''
There are 4 types of Data, namely Discrete, Continuous, Nominal and Ordinal. For categorical data, we will be dealing with Nominal and Ordinal.
We will be encoding categories using 3 different methods.
- ordinal grouping (for ordinal category)
- Get dummies (for categories with less than 8(upto you) unique items)
- Top (8-10) freq occuring item within a category
'''

## Lets create a dataframe of category with its unique features

unq_col = dict()
for col in cate_feat:
    unq_col[col] = list(df[col].unique())

unq_df = pd.DataFrame.from_dict(unq_col, orient="index").replace({None:0})
print(unq_df)
unq_df.to_csv(path+'/data_explore/columns_unq_features.csv', index=True)

## Deealing with the Ordinal var first, These rep some kind of order
cate1 = ["BsmtCond"]
cate1_item = ['none',"Po", "Fa", "TA", "Gd"]

cate2 = ["BsmtExposure"]
cate2_item = ['none','No','Mn','Av','Gd']

cate3 = ["BsmtQual"]
cate3_item = ['none',"Fa","TA","Gd", "Ex"]

cate4 = ["ExterCond", "HeatingQC"]
cate4_item = ["Po", "Fa", "TA", "Gd", "Ex"]

cate5 = ["ExterQual", "KitchenQual"]
cate5_item = ["Fa", "TA", "Gd", "Ex"]

cate6 = ["GarageQual", "GarageCond"]
cate6_item = ['none',"Po", "Fa", "TA", "Gd", "Ex"]

cate7 = ["BsmtFinType1", "BsmtFinType2"]
cate7_item = ['none',"Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]

cate = [cate1,cate2,cate3,cate4,cate5,cate6,cate7]
cate_item = [cate1_item,cate2_item,cate3_item,cate4_item,cate5_item,cate6_item,cate7_item]

# Takes the items in a category and converts into numerical value ( with order respected)
for idx in range(len(cate)):
    encoder = OrdinalEncoder(categories=[cate_item[idx]])

    for col in cate[idx]:
        df[col] = encoder.fit_transform(df[[col]])

cate_ord = cate1+cate2+cate3+cate4+cate5+cate6+cate7

cate_one_hot = list()
cate_target_var = list()

for col in df[cate_feat].drop(cate_ord, axis = 1).columns:
    if len(df[col].unique()) < 6:
        cate_one_hot.append(col)
    else:
        cate_target_var.append(col)

dummies_one_hot = pd.get_dummies(df[cate_one_hot], drop_first = True)
print(dummies_one_hot)


## Finds the top 6 item in each category and creates a one hot encoding. Done on categories with alot
## of items to reduce the dimensions.
def one_hot(df):
    for col in df:
        top_10 = [item for item in df[col].value_counts().sort_values(ascending=False).head(6).index]

        for label in top_10:
            df[label] = np.where(df[col] == label, 1, 0)
    return df


# One hot encoding for nominal categorical data
df_tar_var = one_hot(df[cate_target_var])
df_tar_var.drop(cate_target_var,axis = 1,inplace = True)

# creating a dataframe with all converted values for prediction
df_final = pd.concat([df_tar_var,dummies_one_hot,df],axis = 1)
df_final.drop(cate_feat+cate_ord,axis = 1, inplace=True)

print(df_final.head())

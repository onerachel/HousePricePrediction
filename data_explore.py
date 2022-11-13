import os
path = "/Users/lj/ML_Python/HousePrice"
for dirname, _, filenames in os.walk(path+'/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={"figure.figsize": (20, 15)})
sns.set_style("whitegrid")



# Read files
df_train = pd.read_csv(path+"/data/train.csv")
df_test = pd.read_csv(path+"/data/test.csv")
submission = pd.read_csv(path+"/data/sample_submission.csv")
df = pd.concat([df_train,df_test],axis = 0,ignore_index = True)

## Create a dictionary to see the description of the column names
with open(path+"/data/data_description.txt", "r") as f:
    texts = f.readlines()

newlist = list()
for col in df.columns:
    for text in texts:
        if col in text:
            newlist.append(text.split(":"))

desc = dict()
for item in newlist:
    if len(item) == 2:
        desc[item[0]] = item[1]


## Example
# print(desc["YearRemodAdd"])

# Explore Dataset
## df_train.head(10).style.background_gradient(cmap="viridis") in jupyter
f = open(path+"/data_explore/train_dataset.html",'w')
f.write(df_train.style.background_gradient(cmap="viridis").render())
f.close()

df.info()
print(df.describe().shape) # filter out the object dtype

f = open(path+"/data_explore/complete_dataset.html",'w')
f.write(df.describe().transpose().style.background_gradient(cmap="magma").render())
f.close()

print(df_train.shape)
print(df_test.shape) # the SalePrice column is missing in test data, which we need to predict.

## 1.pair plot
# var_num = ["SalePrice", "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
# p = sns.pairplot(df_train[var_num])
# p.savefig(path+"/data_explore/partial_features.png")
# plt.show()

# 2. simple distribution plot
sns.distplot(df["SalePrice"])
plt.savefig(path+"/data_explore/sale_price.png")
plt.show()

print(df["SalePrice"].describe())

# 3. log distribution plot
df["LogSalePrice"] = np.log10(df["SalePrice"])
sns.distplot(df["LogSalePrice"],color = 'r')
plt.show()
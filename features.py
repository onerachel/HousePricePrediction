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


## Question 1: What numerical features correlates the most with the Sale Price of the house?

# Read files
df_train = pd.read_csv(path+"/data/train.csv")
df_test = pd.read_csv(path+"/data/test.csv")
submission = pd.read_csv(path+"/data/sample_submission.csv")
df = pd.concat([df_train,df_test],axis = 0,ignore_index = True)
df["LogSalePrice"] = np.log10(df["SalePrice"])
## Lets create a list of column names for categorical and numerical features

cate_feat = list(df.select_dtypes(include=[object]).columns)
num_feat = list(df.select_dtypes(include=[int,float]).columns)

print(cate_feat)
print('\n')
print(num_feat)

# Heatmap for all the remaining numerical data including the taget 'SalePrice'
# Define the heatmap parameters
pd.options.display.float_format = "{:,.2f}".format

# Define correlation matrix
corr_matrix = df[num_feat].corr()

# Replace correlation < |0.3| by 0 for a better visibility
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0

# plot the heatmap
sns.heatmap(corr_matrix, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot_kws={"size": 9, "color": "black"},annot=True)
plt.title("SalePrice Correlation")
plt.savefig(path+"/data_explore/heat_map.png")
plt.show()

## Lets visualize individually

corr = df.corr()["SalePrice"].sort_values(ascending=False)[2:8] ## selecting 6 cols other than Saleprice, LogPrice
print(corr)

f, ax = plt.subplots(nrows=6, ncols=1, figsize=(20, 40))
for i, col in enumerate(corr.index):
    sns.scatterplot(x=col, y="SalePrice", data=df, ax=ax[i], color='darkorange')
    ax[i].set_title(f'{col} vs SalePrice')
plt.savefig(path + "/data_explore/correlation_scatter_plot.png")
plt.show()

## What year were most of the houses built (Top 10), and does the year built say anything regarding the sale price?
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=df,)
fig.axis(ymin=0, ymax=900000);
plt.xticks(rotation=90);
plt.tight_layout()
plt.show()

yr_built = pd.DataFrame({"Count":df["YearBuilt"].value_counts()[:10]}).reset_index()
yr_built.rename(columns={'index':'Year'},inplace=True)
plt.figure(figsize = (20,10))
sns.barplot(x = 'Year', y = "Count", data = yr_built)
plt.title("Year Built")
plt.show()

## As a side question, lets see if there is a huge difference in sale price based on different months
df.groupby("MoSold").mean()["SalePrice"].sort_values(ascending=False).plot(kind='bar')
plt.show()


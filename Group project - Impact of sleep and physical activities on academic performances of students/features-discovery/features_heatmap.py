import pandas as pd
import matplotlib.pyplot as plt

# Read Polished data set
df = pd.read_csv("../dataset/_old_polished.csv")
df.head()
df.to_numpy()

targetcolumn = (df.columns.get_loc("dmscore"))

# Define features and target column
X = df.drop(["dmscore"], axis=1)
y = df.iloc[:, targetcolumn]

# Plot heat map
import seaborn as sns

# get correlations of each features in dataset
y = y.to_frame()
for i in range(X.shape[1]):
    while i + 9 <= X.shape[1]:
        Xmod = X.iloc[:, i:i + 9]
        fulldata = pd.concat([Xmod, y], axis=1)
        corrmat = fulldata.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(20, 20))
        g = sns.heatmap(fulldata[top_corr_features].corr(), annot=True, cmap="RdYlGn")
        X = X.drop(columns=Xmod.columns, axis=1)

# plot heatmaps
plt.show()
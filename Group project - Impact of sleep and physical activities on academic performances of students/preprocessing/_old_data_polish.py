import pandas as pd
import numpy as np
import math

df=pd.read_csv("../dataset/2006SIATeensSleepRawData.csv", header=None)


# Drop all columns that are not numerical values
notint = []
for i in range(len(df.columns)):
    try:
        if(math.isnan(float(df[i][1]))):
            notint.append(i)
            pass
    except:
        try:
            notint.append(i)
        except:
            pass
        pass


df = df.drop(notint, axis=1)
df= df.reset_index(drop=True)
df.columns = range(df.shape[1])
print(df)

# Check which questions had an high % of invalid ansers and drop them
todrop=[]
for i in range(len(df.columns)):
    unusable = 0
    for j in range(1,len(df[0])):
        try:
            if(math.isnan(float(df[i][j]))):
                df[i][j]=99
                unusable+=1
            else:
                if( float(df[i][j])>90 and float(df[i][j])< 100) or ( float(df[i][j])>990 and float(df[i][j])< 1000) :
                    unusable+=1
        except:
            unusable+=1
    if(unusable>((len(df[0])/100)*1)):
        todrop.append(i)


# Drop the civil registration columns
for i in range(7):
    todrop.append(i)
todrop.append(169)
todrop.append(170)
todrop.append(171)
df = df.drop(todrop, axis=1)
df= df.reset_index(drop=True)
df.columns = range(df.shape[1])

# Save the polished file
df.to_csv("../dataset/_old_polished.csv",index=False)

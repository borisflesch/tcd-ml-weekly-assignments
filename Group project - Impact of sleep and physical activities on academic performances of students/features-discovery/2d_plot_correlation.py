import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Questionnaire - https://www.sleepfoundation.org/wp-content/uploads/2018/10/SIAQuestionnaire2006.pdf?x65960
# Report - https://www.sleepfoundation.org/wp-content/uploads/2018/10/2006_summary_of_findings.pdf?x65960

df = pd.read_csv("../dataset/2006SIATeensSleepRawData.csv")

# C1 = Time to go to sleep (school night)
# C2 = School night activities habits
# C3 = How long to fall asleep (school night)
# C4 = Get up time (school days)
# C6 = How long do you usually sleep on a normal school night?

# Questions about taking a nap

# P20 = Most got grades

### -> C6(X) VS P20(Y)
# X = df.iloc[:,128]
# y = df.iloc[:,86]
#X2 = df.iloc[:,1]
#X = np.column_stack((X1,X2))


### -> C1(X) VS P20(Y)
# X = df.iloc[:,109]
# y = df.iloc[:,86]

### -> C2_mean(X) VS P20(Y)
# X = df.iloc[:,112:118].mean(axis=1)
# X = df.iloc[:,112] # Did homework or studied — no impact
# X = df.iloc[:,113] # Watched TV - no impact
# X = df.iloc[:,114] # Talked on the phone - no impact
# X = df.iloc[:,115] # Instant messages / internet - no impact
# X = df.iloc[:,116] # Read for fun - Small impact?
# X = df.iloc[:,117] # Played electronic or video games - Small impact?

# X = df.iloc[:,118] # Exercised - Impact
# y = df.iloc[:,86]


### [activity/habit] vs Sleep time
y = df.iloc[:,128] # Sleep time
# X = df.iloc[:,115] # Instant messages / internet
X = df.iloc[:,118] # Exercised - Impact

# y = df.iloc[:, df.columns.get_loc("C6")]
# X = df.iloc[:, df.columns.get_loc("p26")]

print(X.head())
print(y.head())

# remove unanswered
for i in range(X.size):
    if X[i] == 99 or X[i] == 98 or y[i] == 99 or y[i] == 98:
    # if X[i] > 15 or y[i] == 99 or y[i] == 98:
        X.pop(i)
        y.pop(i)

# print(X.size)
# print(y.size)
# plt.plot(X, np.where(y < 5, 0, 1), '+')
plt.plot(X, y, '+')
plt.show()
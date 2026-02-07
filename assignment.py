import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

filename = 'Flight_Delays_2018.csv'

# ARR_DELAY is the column name that should be used as dependent variable (Y).
df = pd.read_csv(filename)

# DROP NULLS
df.dropna(inplace=True)

# Choose independant variables
# Set up x and y for linear regression
y = df["ARR_DELAY"]
x = df[["DEP_DELAY", "CARRIER_DELAY", "LATE_AIRCRAFT_DELAY"]]

# Run the correlation matrix
corr_matrix = df.corr(numeric_only=True).round(2)
print(corr_matrix)

#Find the difference between the airline carriers with scatterplot
print(df.groupby("OP_CARRIER_NAME")["ARR_DELAY"].agg(["max","min","mean"]))
df.plot.scatter(x="OP_CARRIER_NAME",y="ARR_DELAY")
plt.xticks(rotation=45)
plt.show()

#run linear regression model
model = sm.OLS(y, x).fit()
print(model.summary())

#correlation heatmap
variables = pd.concat([x, y], axis=1)
corr = variables.corr()
sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="vlag")
plt.show()

#pairplot
sns.pairplot(variables)
plt.show()

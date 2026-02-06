import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("Dataset\loan_approval_dataset.csv")
df.head()
df.describe()
df.info()
df.columns
sns.pairplot(df)
df.hist(bins=10, figsize=(15,12))

# Bar chart
df[' loan_status'].value_counts().plot(kind='bar')

# Scatter plot

cols = [' income_annum', ' loan_amount', ' loan_term', ' cibil_score',
       ' residential_assets_value', ' commercial_assets_value',
       ' luxury_assets_value', ' bank_asset_value']

for i in cols:
    plt.figure(figsize=(5,3))
    sns.scatterplot(data=df, x=' loan_status', y=i, hue=' loan_status')
    plt.title(f'Scatter plot of {i} vs Loan Status') # Add title
    plt.xlabel('Loan Status') # Add x-label
    plt.ylabel(i)
    plt.legend(title='Loan Status') # Add legend
    plt.show()

# Heat Map
df_copy = df.select_dtypes(include='number')

corr = df_copy.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Box Plot
data = [df[' income_annum'],df[ ' loan_amount'],df[ ' loan_term'],df[ ' cibil_score'],
       df[' residential_assets_value'],df[ ' commercial_assets_value'],
       df[' luxury_assets_value'],df[ ' bank_asset_value']]

plt.figure(figsize=(18,8))
plt.boxplot(data,tick_labels=['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
       'residential_assets_value', 'commercial_assets_value',
       'luxury_assets_value', 'bank_asset_value'])
plt.title("Feature Distribution Comparison")
plt.ylabel("Value")
plt.show()

# Handling missing values
print("Number of missing values per column", df.isna().sum())
print("Total missing values: ",df.isna().sum().sum())
# df.replace(['NA', '?', -1], np.nan, inplace=True) #This converts the missing value as NAN

df[' income_annum'] = df[' income_annum'].fillna(df[' income_annum'].median())

#Capping
num_col = [' income_annum', ' loan_amount']
for col in num_col:
    lower = df[col].quantile(0.1)
    upper = df[col].quantile(0.95)
    df[col] = np.clip(df[col], lower, upper)

# Identifing and handling duplicates
# Before
print("Duplicates (ignoring loan_id):",
      df.drop(columns=['loan_id']).duplicated().sum())

# Drop
df = df.drop_duplicates(subset=df.columns.difference(['loan_id']))

# After
print("Duplicates after drop:",
      df.drop(columns=['loan_id']).duplicated().sum())

# Data splitting

loan_X = df.drop(columns=[' loan_amount'])
loan_y = df[' loan_amount']
lX_train, lX_test, ly_train, ly_test = train_test_split(loan_X, loan_y, test_size=0.2,random_state=42)

# Encoding the categorical values

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

lX_train[' education'] = LabelEncoder().fit_transform(lX_train[' education'])
lX_train[' loan_status'] = LabelEncoder().fit_transform(lX_train[' loan_status'])
lX_train[' self_employed'] = LabelEncoder().fit_transform(lX_train[' self_employed'])

# Scaler
from sklearn.preprocessing import RobustScaler
robust_list = [' residential_assets_value',' commercial_assets_value',' bank_asset_value']
std_list = [' income_annum',' loan_amount',' loan_term']
X_scaled = RobustScaler().fit_transform(lX_train)
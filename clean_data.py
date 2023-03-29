import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


# read in the data
loan_data = pd.read_csv('loan_sanction_train.csv')
test_data = pd.read_csv('loan_sanction_test.csv')

# remove the IDs that are unique for each row
loan_data = loan_data.drop('Loan_ID',axis = 1)

# normalize
loan_data['Gender'] = loan_data['Gender'].map({'Male':0, 'Female':1})
loan_data['Married'] = loan_data['Married'].map({'No':0, 'Yes':1})

# 3+ dependents is simplified to 3, turned into numeric column
loan_data['Dependents'] = loan_data['Dependents'].replace(
    to_replace='3+',
    value=3)
# convert column "a" of a DataFrame
loan_data['Dependents'] = pd.to_numeric(loan_data['Dependents'])

loan_data['Education'] = loan_data['Education'].map({'Not Graduate':0, 'Graduate':1})
loan_data['Self_Employed'] = loan_data['Self_Employed'].map({'No':0, 'Yes':1})
# 1 = urban, 2 = suburban, 3 = rural
loan_data['Property_Area'] = loan_data['Property_Area'].map({'Urban':1, 'Suburban':2, 'Rural':3})
loan_data['Loan_Status'] = loan_data['Loan_Status'].map({'N':0, 'Y':1})


# fill NaNs with column means in each column
loan_data = loan_data.fillna(loan_data.median())

print(loan_data.head(15))
print('Sum of Null values:')
print(loan_data.isnull().sum())

loan_data.to_csv('cleaned_data.csv')
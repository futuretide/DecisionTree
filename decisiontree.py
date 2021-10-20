pip install -U scikit-learn
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

df = pd.read_csv(r'C:\Users\HP\Desktop\Python\diabetes.csv', header=0, names=column_names) 

df

feature_columns = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']


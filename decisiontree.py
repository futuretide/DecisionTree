pip install -U scikit-learn
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

df = pd.read_csv(r'C:\Users\HP\Desktop\Python\diabetes.csv', header=0, names=column_names) 

df

feature_columns = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

from sklearn.tree import export_graphviz
get_ipython().system('pip install --upgrade scikit-learn==0.20.3')
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feature_columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pre))

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,special_characters=True, feature_names = feature_columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

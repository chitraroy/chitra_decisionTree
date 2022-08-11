# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 22:16:28 2022

@author: chitr
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# =============================================================================
# Load the data (student-por.csv) into a pandas dataframe named data_firstname where first name is
# you name.
# =============================================================================
df_chitra = pd.read_csv('C:/Users/chitr/OneDrive/Desktop/Machine_learning/Decision-Tree/student-por.csv', delimiter = ';')
df_chitra



# =============================================================================
# Carryout some initial investigations:
# a. Check the names and types of columns.
# b. Check the missing values.
# c. Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
# d. Check the categorical values.
# e. In you written response write a paragraph explaining your findings about each column.
# Pre-process and prepare the data for machine learning
# =============================================================================


df_chitra.shape


df_chitra.columns


df_chitra.isnull().sum()


df_chitra.describe()








# =============================================================================
# 3. Create a new target variable i.e. column name it pass_fristname, which will store the following per
# row:
# a. 1 : if the total of G1, G2, G3 is greater or equal to 35
# b. 0 : if the total of G1, G2, G3 is less than 35
# 4. Drop the columns G1, G2, G3 permanently.
# 
# 
# =============================================================================


target_column_values = []
for a,b,c in zip(df_chitra['G1'], df_chitra['G2'], df_chitra['G3']):
  if a + b + c >= 35:
    target_column_values.append(1)
  else:
    target_column_values.append(0)
df_chitra.insert(column='pass_chitra', value=target_column_values, loc=0)
print("inserted pass_chitra column = " , df_chitra.columns)
df_chitra.drop(columns=['G1', 'G2', 'G3'], inplace=True)

print(df_chitra.columns)

# =============================================================================
# 
# Separate the features from the target variable (class) and store the features into a dataframe named
# features_first name and the target variable into a dataframe named target_variable_firstname.
# 6. Print out the total number of instances in each class and note into your report and explain your
# findings in terms of balanced and un-balanced.
# =============================================================================


features_chitra = df_chitra.drop('pass_chitra', axis=1)
print(features_chitra.columns)
print(features_chitra)

target_variable_chitra=df_chitra['pass_chitra']
print(target_variable_chitra)

# my data is balanced
import seaborn as sns
sns.countplot(x = 'pass_chitra', data = df_chitra)






# ==========================================

#target_variable_chitra.value_counts

# =============================================================================
# Create two lists one to save the names of your numeric fields and on to save the names of your
# categorical fields. Name the lists numeric_features_firstname and cat_features_firstname
# respectively. To build the lists refer to the documentation https://pandas.pydata.org/pandas-
# docs/stable/reference/api/pandas.DataFrame.select_dtypes.html , be very careful what options you
# select and you don’t miss any columns.
# =============================================================================

numeric_features_chitra = features_chitra.select_dtypes(include=['int64'])
print('numeric value columns',numeric_features_chitra)


cat_features_chitra = features_chitra.select_dtypes(exclude=['int64'])
print("categorical value columns",cat_features_chitra)


# =============================================================================
# Prepare a column transformer to handle all the categorical variables and convert them into numeric
# values using one-hot encoding. The transformer must preserve the rest of the columns. Refer to the
# following documentation https://scikit-
# learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html and carefully check
# all the parameters. Name the transformer transformer_firstname.
# 9. Prepare a classifier decision tree model i.e. an estimator name it clf_firstname, set the
# criterion="entropy"and max_depth = 5. Refer to the documentation https://scikit-
# learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html and check all possible
# parameters.
# =============================================================================


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer_chitra = ColumnTransformer([('One_Hot_Enconder', OneHotEncoder(),list(cat_features_chitra.columns))])


from sklearn.tree import DecisionTreeClassifier

clf_chitra= DecisionTreeClassifier(criterion="entropy", max_depth = 5, min_samples_split=2, min_samples_leaf=1)


# =============================================================================
# Build a pipeline name it pipeline_first_name. The pipeline should have two steps the first the
# column transformer you prepared in step 8 and the second the model you prepared in step 9.
# 
# =============================================================================

from sklearn.pipeline import Pipeline

pipeline_chitra = Pipeline([('column_Transformer', transformer_chitra),
                           ('classifier', clf_chitra)])


# =============================================================================
# 
# Split your data into train 80% train and 20% test, use the last two digits of your student number for
# the seed. Name the train/test dataframes as follows : X_train_firstname, X_test firstname, y_train
# firstname, y_test firstname.
# =============================================================================




from sklearn.model_selection import train_test_split

X_train_chitra, X_test_chitra, y_train_chitra, y_test_chitra = train_test_split(features_chitra, target_variable_chitra , test_size=0.2, random_state=74)

# =============================================================================
#  Fit the training data to the pipeline you built in step #11.
# =============================================================================

clf_chitra_pipeline = pipeline_chitra.fit(X_train_chitra, y_train_chitra)

print(clf_chitra_pipeline)

preds = clf_chitra_pipeline.predict(X_test_chitra)
preds

# =============================================================================
# 
# Cross validate the output on the training data using 10-fold cross validation and use the last two
# digits of your student ID as seed and set the shuffle to True.
# =============================================================================


from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

cross_validations = KFold(n_splits=10, shuffle=True, random_state=74)
scores = cross_validate(clf_chitra_pipeline, features_chitra, target_variable_chitra, cv=cross_validations)

scores['test_score'].mean()

# =============================================================================
# 
# Visualize the tree using Graphviz.
# Note: If Graphviz is not installed please use an anaconda command prompt to install using:
# conda install graphviz python-graphviz
# Then make sure to add the path to the graphviz binaries to your environmental variables. On
# windows these paths could be:
# C:\Anaconda3\envs\env_name\Library\bin\graphviz or C:\Anaconda3\Library\bin\graphvi
# z
# =============================================================================



import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import tree# # 
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(clf_chitra,filled = True);
fig.savefig('./Decisiontree11.png')


# =============================================================================
#  Print out two accuracy score one for the model on the training set i.e. X_train, y_train and the other
# on the testing set i.e. X_test, y_test. Record both results in your written response.
# 19. Use the model to predict the test data and printout the accuracy, precision and recall scores and the
# confusion matrix. Note the results in your written response
# =============================================================================




print("Training accuracy = ", clf_chitra_pipeline.score(X_train_chitra, y_train_chitra))
print("Testing accuracy = ", clf_chitra_pipeline.score(X_test_chitra, y_test_chitra))



from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
y_actual = y_test_chitra
print(y_actual)
y_pred = clf_chitra_pipeline.predict(X_test_chitra)
print(y_pred)

print("Accuracy = ", accuracy_score(y_pred, y_actual))
print("Precision = ", precision_score(y_pred, y_actual))
print("Recall = ", recall_score(y_pred, y_actual))
print("F1 Score = ", f1_score(y_pred, y_actual))
print("Confusion Matrix:")
print(confusion_matrix(y_pred, y_actual))



# =============================================================================
# 0. Using Randomized grid search fine tune your model using the following set of parameters
# parameters= parameters={'m min_samples_split' : range(10,300,20),'m max_depth':
# range(1,30,2),'m min_samples_leaf':range(1,15,3)}
# For the randomized grid search object set the following parameters:
# a. estimator= pipeline_first_name
# b. param_grid= pipeline_first_name
# c. scoring='accuracy'
# d. param_distributions=parameters
# e. cv=5
# f. n_iter = 7 (Number of parameter settings that are sampled. n_iter trades off runtime vs
# quality of the solution.)
# g. refit = True
# h. verbose = 3
# =============================================================================


from sklearn.model_selection import RandomizedSearchCV

parameters= parameters={'classifier__min_samples_split' : range(10,300,20),'classifier__max_depth':range(1,30,2),'classifier__min_samples_leaf':range(1,15,3)}
grid_search_chitra = RandomizedSearchCV(estimator = pipeline_chitra,
                          param_distributions=parameters,
                          scoring='accuracy', 
                          cv=5,
                          n_jobs = 7, 
                          verbose = 3, 
                          refit=True)



# =============================================================================
# Fit your training data to the gird search object
# 
# =============================================================================
grid_search_chitra.fit(X_train_chitra, y_train_chitra)

# =============================================================================
# Print out the best parameters and note them it in your written response.
# Printout the best estimator and note it in your written response
# =============================================================================

grid_search_chitra.best_params_

best_model_chitra = grid_search_chitra.best_estimator_



print("Training accuracy = ", best_model_chitra.score(X_train_chitra, y_train_chitra))
print("Testing Accuracy = ", best_model_chitra.score(X_test_chitra, y_test_chitra))



y_pred_grid = best_model_chitra.predict(X_test_chitra)

print("Accuracy = ", accuracy_score(y_pred_grid, y_actual))

# =============================================================================
# Printout the precision, re_call and accuracy. Compare them with earlier readings you generated
# during steps 20. Are the better or worse explain why.
# 
# =============================================================================


print("Precision = ", precision_score(y_pred_grid, y_actual))
print("Recall = ", recall_score(y_pred_grid, y_actual))
print("F1 Score = ", f1_score(y_pred_grid, y_actual))
print("Confusion Matrix:")
print(confusion_matrix(y_pred_grid, y_actual))


# =============================================================================
# 27. Save the model using the joblib (dump). Note the type should be .pkl
# 28. Save the full pipeline using the joblib – (dump).
# 
# =============================================================================

import joblib
#best_model_chitra = grid_search_chitra.best_estimator_
joblib.dump(best_model_chitra, 'best_model_chitra.pkl', compress = 1)

joblib.dump(clf_chitra_pipeline, 'pipeline_chitra.pkl', compress = 1)



















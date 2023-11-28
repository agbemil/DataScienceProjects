# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:04:00 2023

@author: Emil Agbemade 
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


# Reading the data
df = (pd.read_csv(r"D:\Second Yeaar\Data_Minning II\Homeworks\dat.csv"))

#CHecking for missing values
len(df.loc[(df['ca']=='?')
           |
           (df['thal']=='?')])
#### 6 rows has missing values
df.loc[(df['ca']=='?')
           |
           (df['thal']=='?')]

#Selecting rows with no missing values
df_no_missing = df.loc[(df['ca']!='?')
           &
           (df['thal']!='?')]
len(df_no_missing)

#df_no_missing['ca'].unique()


#Variable selection
target = df_no_missing["num"]
features = df_no_missing.iloc[:,0:13]

###One-hot encoding
X = pd.get_dummies(features,columns=['cp','restecg','slope','thal'])

####Need to recode y
target_not_zero = target >0
target[target_not_zero] = 1
#target.unique

#Preliminary classification tree
#Split data
X_train, X_test, y_train, y_test = train_test_split(
     X, target, random_state=42)


#Fit model
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)


#tree.plot_tree(clf)
fig = plt.figure(figsize=(15,7.5))
tree.plot_tree(clf, rounded = True,
                   feature_names= X.columns,  
                   filled=True, class_names=["No HD", "Yes HD"])
#Confusion Matrix
plot_confusion_matrix(clf,X_test,y_test,display_labels=["Does not have HD","Has HD"])


#Tunning for alpha(Cost Complexity Pruning)
path = clf.cost_complexity_pruning_path(X_train, y_train)#determine values for alpha
ccp_alphas = path.ccp_alphas #extract different values for alpha
ccp_alphas = ccp_alphas[:-1] #Exclude max



clfs = []
#Creating one decission tree per value for alpha and store it in array
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
    
#Ploting the accuracy of the trees
train_score = [clf.score(X_train, y_train) for clf in clfs] 
test_score = [clf.score(X_test, y_test) for clf in clfs] 
train_score 
test_score  

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing")
ax.plot(ccp_alphas,train_score,marker = "o",label = "train",drawstyle = "steps-post")
ax.plot(ccp_alphas,test_score,marker = "o",label = "test",drawstyle = "steps-post")
ax.legend()
plt.show()

#the graph suggest alpha = 0.028

# ==========================================================
 clf = DecisionTreeClassifier(random_state=0,ccp_alpha=0.028)
 
# #We then use 5-fold crossvalidation
 scores = cross_val_score(clf,X_train,y_train,cv = 5)
 dat = pd.DataFrame(data={'tree': range(5),'accuracy': scores})
 
 dat.plot(x = 'tree', y='accuracy',marker='o',linestyle='--')
 

 alpha_loop_values = []
 
 for ccp_alpha in ccp_alphas:
     clf = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
     scores = cross_val_score(clf,X_train,y_train,cv = 5)
     alpha_loop_values.append([ccp_alpha,np.mean(scores),np.std(scores)])
 
# #Plotting graphs for means and alpha
 alpha_result = pd.DataFrame(alpha_loop_values,columns=['alpha','mean_accuracy','std'])
 alpha_result.plot(x = 'alpha',
                   y = 'mean_accuracy',
                   yerr = 'std',
                   marker = 'o',
                   linestyle = '--')
 
# #Looking for the exact value
 
 alpha_result[(alpha_result['alpha']>  0.012)
              &
              (alpha_result['alpha']< 0.015)]
 
 
# #Now lets store the ideal value for alpha so that we can use it to build the best tree
 ideal_ccp_alpha = alpha_result[(alpha_result['alpha']> 0.012)
              &
              (alpha_result['alpha']< 0.015)]['alpha']
 
 ideal_ccp_alpha
 
# #Convert to float
 ideal_ccp_alpha=float(ideal_ccp_alpha)
 ideal_ccp_alpha
# =================================================

#Now we have ideal alpha hence we can buil and evaluate the final classification
clf_pruned = DecisionTreeClassifier(random_state=0,ccp_alpha=0.0142)
clf_pruned= clf_pruned.fit(X_train, y_train)

####Final Confusion matrix
plot_confusion_matrix(clf_pruned,
                      X_test,
                      y_test,
                      display_labels=["Does not have HD","Has HD"])


#####Ploting our pruned tree
fig = plt.figure(figsize=(15,7.5))
tree.plot_tree(clf_pruned, rounded = True,
                   feature_names= X.columns,  
                   filled=True, class_names=["No HD", "Yes HD"])














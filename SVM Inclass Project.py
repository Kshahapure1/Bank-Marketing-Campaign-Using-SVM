
# SVM classification
# dataset: bank-additional-full.csv

# libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder 

# visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import pylab

# read the file

data = pd.read_csv("bank-additional-full.csv",delimiter=(";"))

data.shape
data.head()
data.tail()


data.columns
# since 'contact' is not relivent feature, remove this from the dataset
data.drop(columns='contact',inplace=True)
diab.columns

# check the y-distribution
data["y"].value_counts()

#EDA

#Split features into numeric and factor columns
def splitcols(data):
    nc=data.select_dtypes(exclude='object').columns.values
    fc=data.select_dtypes(include='object').columns.values
    
    return(nc,fc)

nc,fc = splitcols(data)
nc
fc


#Check for nulls

data.isnull().sum() #No nulls present

#Cheeck for the zeros

for c in data:
    print(c, " == ",data[c][data[c]==0])
    
#duration
data.duration.describe()
data.duration[data.duration == 0] = np.mean(data.duration)

#pdays
data.pdays.describe()
data.pdays[data.pdays == 0] = np.mean(data.pdays)

#previous
data.previous.describe()  
data.previous[data.previous == 0] = np.mean(data.previous)


#Check for Collinearity

def plotdata(data,nc,ctype):
    if ctype not in ['h','c','b']:
        msg='Invalid Chart Type specified'
        return(msg)
    
    if ctype=='c':
        cor = data[nc].corr()
        cor = np.tril(cor)
        sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,
                    yticklabels=nc,square=False,annot=True,linewidths=1)
    else:
        COLS = 2
        ROWS = np.ceil(len(nc)/COLS)
        POS = 1
        
        fig = plt.figure() # outer plot
        for c in nc:
            fig.add_subplot(ROWS,COLS,POS)
            if ctype=='b':
                sns.boxplot(data[c],color='yellow')
            else:
                sns.distplot(data[c],bins=20,color='green')
            
            POS+=1
    return(1)

plotdata(data,nc,"c")

#emp.var.rate and euribor3m showing higher correlation so, remove these columns
#make a copy of dataset

data1= data.copy()

data1.drop(columns = ["emp.var.rate","euribor3m"],inplace=True)

len(data.columns)
len(data1.columns)

#split data1

nc,fc = splitcols(data1)

#Check for Outliers
plotdata(data1,nc,"b") # There are no such outliers persent in the daataset 
nc
fc

#To check unique values in each feature 

for c in fc:
    print(c ," === " ,data1[c].unique())
    
   
le=LabelEncoder()
for i in fc:   
    data1[i]= le.fit_transform(data1[i])
    print(data1)

#refres the split featires
nc,fc = splitcols(data1)
nc
fc

# standardize the data
def stdData(data,y,std):
    
    D = data.copy()
    
    if std == "ss":
        tr = preprocessing.StandardScaler()
    elif std == "minmax":
        tr = preprocessing.MinMaxScaler()
    else:
        return("Invalid Type specified")
    
    D.iloc[:,:] = tr.fit_transform(D.iloc[:,:])
    
    # restore the actual y-value
    D[y] = data[y]
    
    return(D)

# standard scaler data
data_ss = stdData(data1,"y","ss")

# minmax scaler data
data_mm = stdData(data1,"y","minmax")

data_ss.head()
data_mm.head()



# split the stdscaler data into train and test
trainx,testx,trainy,testy = train_test_split(data_ss.drop("y",1),
                                                 data_ss["y"],test_size=0.30)

print(trainx.shape,trainy.shape)
print(testx.shape,testy.shape)


# split the minmaxscaler data into train and test
trainx1,testx1,trainy1,testy1 = train_test_split(data_mm.drop("y",1),
                                                 data_mm["y"],test_size=0.30)


print(trainx1.shape,trainy1.shape)
print(testx1.shape,testy1.shape)


# SVM specific parameters

# list of values for C and gamma
lim=3
lov_c = np.logspace(-1,1,lim)
lov_g = np.random.random(lim)

'''
kernels in SVM
linear -> C
sigmoid -> C,gamma
poly -> C,gamma
rbf(radial basis function) -> C,gamma
'''

# build the parameters
params = [{ 'kernel':['linear'], 'C':lov_c, 
            'kernel':['sigmoid'],'C':lov_c,'gamma':lov_g,
            'kernel':['poly'],'C':lov_c,'gamma':lov_g,
            'kernel':['rbf'],'C':lov_c,'gamma':lov_g}]

# perform Grid Search
model = svm.SVC()

# grid CV on stdscaler data
grid = GridSearchCV(model,param_grid=params,
                    scoring="accuracy",cv=2,
                    n_jobs=-1).fit(trainx,trainy)

# best parameters
bp = grid.best_params_
bp1 = bp.copy()
bp1

# build the model with the best parameters
m1 = svm.SVC(kernel=bp['kernel'], C=bp['C'], gamma=bp['gamma']).fit(trainx,trainy)

# predictions
p1 = m1.predict(testx)

def cm(actual,pred):
    # model accuracy
    print("Model Accuracy = {}".format(accuracy_score(actual,pred)))
    print("\n")
    
    # confusion matrix
    df = pd.DataFrame({'actual':actual,'pred':pred})
    print(pd.crosstab(df.actual,df.pred,margins=True))
    print("\n")
    
    # classification report
    print(classification_report(actual,pred))
    
    return(1)

cm(testy,p1)

# grid CV on MinMaxscaler data
grid = GridSearchCV(model,param_grid=params,
                    scoring="accuracy",cv=2,
                    n_jobs=-1).fit(trainx1,trainy1)

bp2 = grid.best_params_

# build the model with the best parameters
m2 = svm.SVC(kernel=bp2['kernel'], C=bp2['C'], gamma=bp2['gamma']).fit(trainx1,trainy1)

# predictions
p2 = m2.predict(testx1)

cm(testy1,p2)


# balanced sampling
# ------------------
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

perc=0.75
oversamp=SMOTE(sampling_strategy = perc)
undersamp=RandomUnderSampler(sampling_strategy = perc)

steps = [('o',oversamp), ('u',undersamp)]

bsX,bsY = Pipeline(steps=steps).fit_resample(data1.drop('y',1),data1.y)

# create the new dataset
data_tr = bsX.join(bsY)

# compare the 2 datasets (original / balanced sample)
len(data1), len(data_tr)

# compare distribution of classes (original / oversampled)
data1.y.value_counts(), data_tr.y.value_counts()

# split the stdscaler data into train and test
trainx2,testx2,trainy2,testy2 = train_test_split(data_tr.drop("y",1),
                                                 data_tr["y"],test_size=0.30)

print(trainx2.shape,trainy2.shape)
print(testx2.shape,testy2.shape)


# grid CV on Balanced sampled data
grid = GridSearchCV(model,param_grid=params,
                    scoring="accuracy",cv=2,
                    n_jobs=-1).fit(trainx2,trainy2)

# best parameters
bp3 = grid.best_params_


# build the model with the best parameters
m3 = svm.SVC(kernel=bp3['kernel'], C=bp3['C'], gamma=bp3['gamma']).fit(trainx,trainy)

# predictions
p3 = m3.predict(testx2)


#Confusion Matrics
cm(testy2,p3)


'''
CONCLUSION : - 
#Split ratio taken for ec model is 70/30 ratio

Model 1 (M1) = Using Standerd Scelar transformation and Greed Search

              precision    recall  f1-score   support

           0       0.91      0.98      0.94     10932
           1       0.61      0.24      0.34      1425

    accuracy                           0.89     12357
   macro avg       0.76      0.61      0.64     12357
weighted avg       0.87      0.89      0.87     12357

Model 2 (M2) = Using MinMax Standerdisation and Greed search 

              precision    recall  f1-score   support

           0       0.92      0.98      0.95     10974
           1       0.70      0.33      0.44      1383

    accuracy                           0.91     12357
   macro avg       0.81      0.65      0.70     12357
weighted avg       0.90      0.91      0.89     12357


Model 3 (M3) = Using Balanced Sampling method and Greed search 

              precision    recall  f1-score   support

           0       0.57      1.00      0.73     11005
           1       0.00      0.00      0.00      8183

    accuracy                           0.57     19188
   macro avg       0.29      0.50      0.36     19188
weighted avg       0.33      0.57      0.42     19188



#We founnd that Model M2 MinMax standerdisation with Greed Search isoptimum model because it has good accuracy, Though all the models has imbalenced in classes prrediction
'''













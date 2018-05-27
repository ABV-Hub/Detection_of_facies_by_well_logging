# coding: utf-8

# LR

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import preprocessing

from pandas import set_option
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

training_data = pd.read_csv('EMDdata.csv', sep='\t')
#training_data = pd.read_csv('datawmTrees.csv', sep='\t')


training_data


training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()
training_data = training_data.drop(['Formation'], axis=1)
training_data = training_data.drop(['Unnamed: 0'], axis=1)


blind = training_data[training_data['Well Name'] == 'KIMZEY A']
training_data = training_data[training_data['Well Name'] != 'KIMZEY A']


#code is borrowed from @kwinkunks
###
facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00',
       '#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                 'WS', 'D','PS', 'BS']
#facies_color_map is a dictionary that maps facies labels
#to their respective colors
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]

def label_facies(row, labels):
    return labels[ row['Facies'] -1]
    
training_data.loc[:,'FaciesLabels'] = training_data.apply(lambda row: label_facies(row, facies_labels), axis=1)
###


correct_facies_labels = training_data['Facies'].values

feature_vectors = training_data.drop(['Well Name', 'Depth','Facies','FaciesLabels'], axis=1)


scaled_features = (feature_vectors - np.mean(feature_vectors))/np.std(feature_vectors)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, correct_facies_labels, test_size=0.2)


from sklearn.linear_model import LogisticRegression


penalty = ['l1','l2']
cc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,3,4,5,6,7,8,9,10,11]
res = np.zeros((len(penalty), len(cc)))
k=0
for p in penalty:
    k1=0
    for j in cc:
        clf = LogisticRegression(C=j, penalty=p)
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        c = 0
        for i in range(len(y_pred)):
            if (y_test[i] == y_pred[i]):
                c = c + 1
        res[k][k1] = c / float(len(y_pred)) 
        k1=k1+1
        #print(res)
    k=k+1


plt.plot(cc,res[0], label = 'l1', color = 'salmon')
plt.plot(cc,res[1], label = 'l2', color = 'skyblue')
plt.title("Selection of parameters")
plt.legend()
plt.xlabel('C')
plt.ylabel('accuracy')


res


correct_facies_labels = training_data['Facies'].values

feature_vectors = training_data.drop(['Well Name', 'Depth','Facies','FaciesLabels'], axis=1)


scaled_features = (feature_vectors - np.mean(feature_vectors))/np.std(feature_vectors)


X_train = scaled_features
y_train = correct_facies_labels


y_blind = blind['Facies'].values


well_features = blind.drop(['Well Name', 'Depth','Facies'], axis=1)


X_blind = (well_features - np.mean(well_features))/np.std(well_features)


clf = LogisticRegression(C=0.8, penalty = 'l1')


clf.fit(X_train, y_train)


res = {}

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_blind).astype(int)
c = 0
for i in range(len(y_pred)):
    if (y_blind[i] == y_pred[i]):
        c = c + 1
res = c / float(len(y_pred)) 
res


y_pred = clf.predict(X_blind)
y_pred = y_pred.astype(int)
blind['Prediction'] = y_pred


#code is borrowed from @kwinkunks
###
def compare_facies_plot(logs, compadre, facies_colors):
    #make sure logs are sorted by depth
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=logs.Depth.min(); zbot=logs.Depth.max()
    
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[compadre].values,1), 100, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                    cmap=cmap_facies,vmin=1,vmax=9)
    
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im2, cax=cax)
    cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 
                                'SiSh', ' MS ', ' WS ', ' D  ', 
                                ' PS ', ' BS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(),logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(),logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(),logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(),logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(),logs.PE.max())
    ax[5].set_xlabel('Facies')
    ax[6].set_xlabel(compadre)
    
    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[6].set_xticklabels([])
    f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14,y=0.94)
###


compare_facies_plot(blind, 'Prediction', facies_colors)

#postprocessing
n = 2

for i in range(n-1, len(y_pred)-n):
    check = 0
    for j in range(n+1):
        if((y_pred[i] != 1) and ((y_pred[i] != y_pred[i-np.asarray(range(n+1))+j]).any())):
            check = check + 1
    if (check == n+1):
        am = [0,0]
        k1 = 1
        while (y_pred[i-1] == y_pred[i-k1]):
            am[0] = am[0] + 1
            if ((i-k1) == 0):
                break
            k1 = k1 + 1
        k2 = n-1
        while (y_pred[i+1] == y_pred[i+k2]):
            am[1] = am[1] + 1
            if (i+k2 != (len(y_pred)-n)):
                k2 = k2 + 1
            else:
                break
        if (am[0] == max(am)):
            y_pred[i] = y_pred[i-1]
        else:
            y_pred[i] = y_pred[i+1]        


blind['Prediction'] = y_pred
compare_facies_plot(blind, 'Prediction', facies_colors)


p = 0

for i in range(1, len(y_pred)):
    if (y_pred[i] == y_blind[i]):
        p = p + 1
accuracy = p/float(len(y_pred))
print('Facies classification accuracy = %f' % accuracy)


Precision = {}
Recall = {}

for j in range(1, 10):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(1, len(y_pred)):
        if ((y_blind[i] == j) and (y_pred[i] == j)):
            TP = TP + 1
        if ((y_blind[i] != j) and (y_pred[i] == j)):
            FP = FP + 1
        if ((y_blind[i] == j) and (y_pred[i] != j)):
            FN = FN + 1
        if ((y_blind[i] != j) and (y_pred[i] != j)):
            TN = TN + 1   
    if (TP+FP == 0):
        Precision[j] = 0
    else:
        Precision[j] = float(TP)/(TP+FP)
    if (TP+FN == 0):
        Recall[j] = 0
    else:
        Recall[j] = float(TP)/(TP+FN)


f={}
for i in range(1, 10):
    if ((Precision[i] == 0) or (Recall[i] == 0)):
        f[i] = 0
    else:
        f[i] = 2*(Precision[i]*Recall[i])/(Precision[i]+Recall[i])
f

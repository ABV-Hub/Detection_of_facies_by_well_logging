
# coding: utf-8

# Заполнение пропусков PE

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pandas import set_option
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None

training_data = pd.read_csv('http://www.people.ku.edu/~gbohling/EECS833/facies_vectors.csv')

training_data


training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()


training_data_wm = training_data[training_data['Well Name'] != 'ALEXANDER D']
training_data_wm = training_data_wm[training_data_wm['Well Name'] != 'KIMZEY A']
training_data_wm = training_data_wm[training_data_wm['Well Name'] != 'Recruit F9']


training_data_m = training_data[training_data['Well Name'] != 'SHRIMPLIN']
training_data_m = training_data_m[training_data_m['Well Name'] != 'SHANKLE']
training_data_m = training_data_m[training_data_m['Well Name'] != 'LUKE G U']
training_data_m = training_data_m[training_data_m['Well Name'] != 'CROSS H CATTLE']
training_data_m = training_data_m[training_data_m['Well Name'] != 'NOLAN']
training_data_m = training_data_m[training_data_m['Well Name'] != 'NEWBY']
training_data_m = training_data_m[training_data_m['Well Name'] != 'CHURCHMAN BIBLE']


X_train = training_data_wm.drop(['PE','Formation', 'Well Name', 'Depth','Facies'], axis=1)
y_train = training_data_wm['PE']


X_test = training_data_m.drop(['PE', 'Formation', 'Well Name', 'Depth','Facies'], axis=1)
y_test = training_data_m['PE']


correct_facies_labels = training_data_wm['PE'].values

feature_vectors = training_data_wm.drop(['Formation', 'Well Name', 'Depth','Facies','PE'], axis=1)


# выбор нормировки

from sklearn import preprocessing

#стандартная
scaler = preprocessing.StandardScaler().fit(feature_vectors)
#scaler = preprocessing.Normalizer().fit(feature_vectors)
scaled_features = scaler.transform(feature_vectors)

#без нормировки
#scaled_features = np.array(feature_vectors)

#преобразование
#scaled_features = (feature_vectors - np.mean(feature_vectors))/np.max(abs(feature_vectors - np.mean(feature_vectors)))
#scaled_features = (feature_vectors - np.mean(feature_vectors))/np.std(feature_vectors)

#нелинейная
#a = 1.2
#minx = np.min(feature_vectors)
#maxx = np.max(feature_vectors)
#xi = (maxx+minx).astype(float)/2
#гиперболический тангенс
#scaled_features = (np.exp(a*(feature_vectors-xi))-1)/((np.exp(a*(feature_vectors-xi))+1))
#сигмоида
#scaled_features = 1/((np.exp((-a)*(feature_vectors-xi))+1)).astype(float)
#


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, correct_facies_labels, test_size=0.2)


y_train = (y_train*1000).astype(int)


from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


models = [LinearRegression(),
          RandomForestRegressor(n_estimators=200, max_features ='sqrt'),
          KNeighborsRegressor(),
          SVR(kernel='linear'),
          LogisticRegression(),
          ]

from sklearn.metrics import r2_score


res = {}
r2 = {}
j = 0
for model in models:   
    j = j + 1
    model.fit(X_train, y_train) 
    pred_pe = model.predict(X_test)
    pred_pe = pred_pe.astype(float)/1000
    r2[j] = r2_score(y_test, pred_pe)
    c = 0
    for i in range(len(pred_pe)):
        c = c + abs(y_test[i] - pred_pe[i])
    res[j] = c / float(len(pred_pe))


md = ['Linear Regression', 'Random Forest Regressor', 'KNeighborsRegressor', 'SVR (linear kernel)', 
      'Logistic Regression']
res1 = {'accuracy': res}
res2 = {'r2': r2}


Res = pd.DataFrame(res1, index = range(1,len(res)+1))
Res['model'] = md


x = range(5)
Res.plot(kind = 'bar', color = 'brown', legend = False)
plt.xticks(x, md)
plt.title('Accuracy')
plt.show()


Res


Res = pd.DataFrame(res2, index = range(1,len(res)+1))
Res['model'] = md


x = range(5)
Res['r2'].plot(kind = 'bar', color = 'skyblue', legend = False)
plt.xticks(x, md)
plt.title('R2')
plt.show()


Res


clf = RandomForestRegressor(n_estimators=200, max_features ='sqrt')
clf.fit(X_train, y_train) 
pred_pe = clf.predict(X_test)


plt.plot(y_test[range(100)], color = 'salmon', label = 'original')
plt.plot(pred_pe[range(100)]/1000, color = 'skyblue', label = 'prediction')
plt.legend()
plt.title("Original PE and prediction")
plt.xlabel('depth (ft)')
plt.ylabel('PE')


# Лучший алгоритм

X_train = training_data_wm.drop(['PE','Formation', 'Well Name', 'Depth','Facies'], axis=1)
y_train = training_data_wm['PE']


X_test = training_data_m.drop(['PE', 'Formation', 'Well Name', 'Depth','Facies'], axis=1)
y_test = training_data_m['PE']


X_train = scaler.transform(X_train)


best_model = models[1]
best_model.fit(scaled_features, y_train)
missing_val = best_model.predict(X_test)


for i in training_data_m:
    training_data_m['PE'] = missing_val


wm = pd.DataFrame(training_data_wm)
m = pd.DataFrame(training_data_m)
ds = wm.append(m)
ds


ds.to_csv('datawmTrees.csv', sep='\t')


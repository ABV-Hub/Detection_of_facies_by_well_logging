
# coding: utf-8



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

training_data = pd.read_csv('datawmTrees.csv', sep='\t')
training_data

training_data = training_data.drop(['Unnamed: 0'], axis = 1)


from PyEMD import EMD

params = ['GR',  'DeltaPHI', 'PHIND', 'PE']
wells = ['SHRIMPLIN', 'ALEXANDER D', 'SHANKLE', 'LUKE G U', 'KIMZEY A', 'CROSS H CATTLE', 'NOLAN', 'Recruit F9', 'NEWBY', 'CHURCHMAN BIBLE']

emd = EMD()

for i in wells:
    for j in params:
        IMFs = emd(np.array(training_data[j][training_data['Well Name'] == i]))
        training_data[j][training_data['Well Name'] == i] -= IMFs[0]

training_data.isnull().any().any()

training_data.to_csv('EMDdata.csv', sep='\t')

IMFs = emd(np.array(training_data[params[0]][training_data['Well Name'] == wells[0]]))

plt.plot(training_data[params[0]][training_data['Well Name'] == wells[0]][range(50,200)], color = 'skyblue', label = 'original')
plt.plot((training_data[params[0]][training_data['Well Name'] == wells[0]]-IMFs[0])[range(50,200)], color ='salmon', label='emd')
plt.legend()
plt.title("EMD")
plt.xlabel('depth (ft)')
plt.xticks(range(50,200),xx, rotation =90)
plt.ylabel('GR (API)')


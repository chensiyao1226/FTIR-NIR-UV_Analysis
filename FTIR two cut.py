import os
import csv
import numpy as np
import pandas as pd
import openpyxl as xl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


path_root = input('input folder root:')
rep_t = 1
#============collect data=================
counter = 0
content_list=np.zeros((3736,2))
content_matrix=[]

for root, dirs, files in os.walk(path_root):
    for filename in files:
        if ('.csv' in filename):
            file_path = os.path.join(root, filename)
            print(filename)
            samp_sheet = pd.read_csv(file_path, skiprows=19, encoding = "shift-jis", header=None)
            samp_dat = samp_sheet.iloc[0:3736, 0:2]
            content_list = samp_dat.values




            if(counter==0):
                content_list[:, 0] = [float(x) for x in content_list[:, 0]]
                content_matrix.append(content_list[:,0])

            content_list[:, 1] = [float(x) for x in content_list[:, 1]]
            content_matrix.append(content_list[:,1])

            counter += 1

#=========cut================
Edge1L = 900; Edge1Lnum=10000 ##minEdge
Edge1H = 1900; Edge1Hnum=100 ##minEdge
Edge2L = 2400; Edge2Lnum=10000 ##maxEdge
Edge2H = 3400; Edge2Hnum=100 ##maxEdge


print(len(content_matrix[0]))

for i in range (len(content_matrix[0])):
    if((content_matrix[0][i]>Edge1L) and (i<Edge1Lnum)):
        Edge1Lnum = i
    if ((content_matrix[0][i] < Edge1H) and (i > Edge1Hnum)):
        Edge1Hnum = i
    if ((content_matrix[0][i] > Edge2L) and (i < Edge2Lnum)):
        Edge2Lnum = i
    if ((content_matrix[0][i] < Edge2H) and (i > Edge2Hnum)):
        Edge2Hnum = i



content_cutted1=[x[Edge1Lnum:Edge1Hnum] for x in content_matrix]
content_cutted1 = np.array(content_cutted1)
content_cutted2=[x[Edge2Lnum:Edge2Hnum] for x in content_matrix]
content_cutted2 = np.array(content_cutted2)

content_cutted=np.append(content_cutted1,content_cutted2,axis=1)
print(content_cutted)
#content_cutted=np.zeros((counter,(Edge2num-Edge2num)))

#content_cutted = np.vstack((content_matrix[0],content_cutted))

plt.figure()
plt.plot(content_cutted[0], content_cutted[1])
plt.show()

#==============Savitzky-Golay=============
from scipy.signal import savgol_filter

savgol=content_cutted
savgol[0] = content_cutted[0]
for i in range(1,content_cutted.shape[0]):
    savgol[i]=savgol_filter(content_cutted[i], window_length = 5, polyorder = 2, deriv = 2)

plt.figure()
plt.plot(savgol[0], savgol[1])
plt.show()

#==============normalization=============
def SNV(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

    return output_data

norm_data = savgol
norm_data[1:] = SNV(savgol[1:])

'''
for i in range(content_cutted.shape[0]-1):
    plt.figure()
    plt.plot(savgol[0], norm_data[i])
    plt.show()
'''

#===========PCA===========

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
pca = PCA(int(counter/rep_t))
pca.fit(norm_data[1:])

# データを主成分空間に写像
features = pca.transform(norm_data[1:])

# 主成分得点
point = pd.DataFrame(features, columns=["PC{}".format(x + 1) for x in range(int(((norm_data.shape[0])-1)/rep_t))])
print(point)


# 第一主成分と第二主成分でプロットする
# np.random.rand(n)产生1*n数组，元素大小0-1
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=list(norm_data[1:, 0]))

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


for i in range(len(features[:, 0])):
    ax.text(features[i, 0],features[i, 1], features[i, 2], i)

plt.show()

'''
plt.figure(figsize=(6, 6))
plt.scatter(features[:, 0], features[:, 1], alpha=0.8, c=list(norm_data[1:, 0]))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
'''


# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()


# PCA の固有ベクトル
Intr_matrix = pd.DataFrame(pca.components_, columns=norm_data[0,:], index=["PC{}".format(x + 1) for x in range(int(((norm_data.shape[0]-1))/rep_t))])
print(Intr_matrix)
'''
from pandas import plotting
plotting.scatter_matrix(pd.DataFrame(features,
                        columns=["PC{}".format(x + 1) for x in range(norm_data.shape[0]-1)]),
                        figsize=(8, 8), c=list(norm_data[1:, 0]), alpha=0.5)
plt.show()
'''


#============transfer================
t1=np.zeros((int(counter/rep_t),norm_data.shape[1]))
t2=np.zeros((int(counter/rep_t),norm_data.shape[1]))
t3=np.zeros((int(counter/rep_t),norm_data.shape[1]))

for i in range(int(counter/rep_t)):
    t1[i] = norm_data[i+1,:]*Intr_matrix.values[0]
    t2[i] = norm_data[i + 1, :] * Intr_matrix.values[1]
    t3[i] = norm_data[i + 1, :] * Intr_matrix.values[2]

    plt.figure()
    plt.plot(norm_data[0,:], t1[i]+t2[i]+t3[i], color='red', alpha=0.6)
    plt.plot(norm_data[0, :], norm_data[i + 1, :], color='green', alpha=0.2)
    plt.show()

print(t1)
print(t2)

R=np.zeros((int(counter/rep_t),norm_data.shape[1]))

for i in range(int(counter/rep_t)):
    R[i] = norm_data[i + 1, :]- t1[i]-t2[i]

    plt.figure()
    plt.plot(norm_data[0, :], R[i], color='red', alpha=0.6)
    plt.plot(norm_data[0, :], norm_data[i + 1, :], color='green', alpha=0.2)
    plt.show()


#=========save==============
'''
try:
    exc_save_magn = path_root + '\\' + 'summary.csv'

    book = xl.Workbook()
    sheet1 = book.create_sheet("original")

    for i in range (counter+1):
        list = content_matrix[i]
        list = list.tolist()
        sheet1.append(list)

    book.save(exc_save_magn)
    print("Saved")

except:
    print("Falied!!!")
'''







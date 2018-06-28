#%%
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.datasets import load_iris

lda = LinearDiscriminantAnalysis(n_components=3)
data = load_iris()

iris_x = data.data
iris_y = data.target
target_names = data.target_names

x_train, x_test, y_train, y_test = train_test_split(iris_x,iris_y, test_size=0.9)


lda_train = lda.fit(x_train,y_train)
lda_prediction = lda.predict(x_test)
lda_ranges = lda.transform(x_test)

end = x_test.shape[0]
colors = ['navy', 'turquoise', 'orange']
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for i in range(end):
    if lda_prediction[i] != y_test[i]:
        color = 'red'
    else:
        color = colors[y_test[i]]

    ax.scatter(x_test[i][0], x_test[i][1],x_test[i][2], c=color)



#print(str(ranges[0][0]) + " " + str(ranges[0][1]))
#print(str(ranges[1][0]) + " " + str(ranges[1][1]))
#print(ranges)



#plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()

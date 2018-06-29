#%%
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_digits()
digit_x = data.data
digit_y = data.target
x_train, x_test, y_train, y_test = train_test_split(digit_x, digit_y, test_size=0.33)

accuracy = []
for x in range(1,50):
    knn = KNeighborsClassifier(x)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    acc = accuracy_score(pred,y_test)
    accuracy.append(acc)

plt.plot([i for i in range(1,50)],accuracy, color='red')
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.show()
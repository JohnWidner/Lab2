#%%
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis(n_components=4)

data = load_iris()
#separate the data and the target
iris_x = data.data
iris_y = data.target

#splits the data into test and train sets. set to  50/50 to get more incorrect results
x_train, x_test, y_train, y_test = train_test_split(iris_x,iris_y, test_size=0.5)

#train the data
lda_train = lda.fit(x_train,y_train)
lr.fit(x_train, y_train)

#array of the predict results
lda_prediction = lda.predict(x_test)
lr_prediction = lr.predict(x_test)

#finds the number of data points to test
end = lda_prediction.shape[0]

#colors for marking thd data on the plot
colors = ['navy', 'turquoise', 'orange']




#plots the results of the LRA
plt.figure(1)
for i in range(end):
    if lda_prediction[i] != y_test[i]:
        color = 'red'
        #print(x_test[i][0], x_test[i][1])
    else:
        color = colors[y_test[i]]
    plt.scatter(x_test[i][0], x_test[i][1], c=color)

#sets the labels
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title('LDA of IRIS dataset "Sepal"')  
    

plt.figure(2)
for i in range(end):    
    if lda_prediction[i] != y_test[i]:
        color = 'red'
        #print(x_test[i][2], x_test[i][3])
    else:
         color = colors[y_test[i]]
    
    plt.scatter(x_test[i][2], x_test[i][3], c=color)

#sets the labels
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title('LDA of IRIS dataset "Petals"')    


#plots the LR results
plt.figure(3)

for i in range(end):
    if lr_prediction[i] != y_test[i]:
        color = 'red'
        #print(x_test[i][0], x_test[i][1])
    else:
        color = colors[y_test[i]]
    plt.scatter(x_test[i][0], x_test[i][1], c=color)

#sets the labels
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title('LR of IRIS dataset "Sepal"')  
    

plt.figure(4)

for i in range(end):    
    if lr_prediction[i] != y_test[i]:
        color = 'red'
        #print(x_test[i][2], x_test[i][3])
    else:
         color = colors[y_test[i]]
    
    plt.scatter(x_test[i][2], x_test[i][3], c=color)

#sets the label
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title('LR of IRIS dataset "Petals"')    

plt.show()


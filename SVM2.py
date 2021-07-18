from sklearn.datasets import make_circles
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x,y=make_circles(n_samples=100)
model=SVC(kernel='rbf',C=4)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
plot_decision_regions(x,y,clf=model)
plt.show()




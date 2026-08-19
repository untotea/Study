from sklearn import datasets
import numpy as np

iris = datasets.load_iris() #Загружаем данные ириса
X = iris.data[:, [2, 3]] #берем длину и ширину лепестка
y = iris.target #Берем метку класса. target в scikit-learn по умолчанию имеет ввиду метки классов.
#print('Метки классов:', np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y) #Случайным образом разбиваем данные на 30% ипсытательных и 70% обучающих.

#print('Количества меток в y:', np.bincount(y))
#print('Количества меток в y_train:', np.bincount(y_train))
#print('Количества меток в y_test:', np.bincount(y_test))

#Стандартизация признаков
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) #StandardScaler оценивает параметры стандартного оклонения и выборочного среднего.
X_train_std = sc.transform(X_train) #Стандартизируем данные.
X_test_std = sc.transform(X_test)

#Обучем модель
from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train) 

y_pred = ppn.predict(X_test_std)
#print('Неправильно классифицированных образцов: %d' % (y_test != y_pred).sum())

#Метрики эффективности
from sklearn.metrics import accuracy_score

#print('Правильность: %.3f' % accuracy_score(y_test, y_pred))
#print('Правильность: %.3f' % ppn.score(X_test_std, y_test))



from matplotlib.colors import ListedColormap #класс из библиотеки для создания ползовательских палитр
import numpy as np
from ML.iris_perceptron.perceptron import Perceptron
import matplotlib.pyplot as plt #запомни, что для методов x/ylabel, scatter и других в конце нужен pyplot
from ML.iris_perceptron.Error_classification_graphic import ppn
from ML.iris_perceptron.iris import *

def plot_decision_regions(X, y, classifier, resolution = 0.02): #Функция принимает значения X, y из iris. classifier - модель классификации.
    #Настроит генератор маркеров и карту цветов
    markers = ('s', 'x', 'o', '4', '4') #создаем маркеры
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') # создаем цвета
    cmap = ListedColormap(colors[:len(np.unique(y))]) #создаем карту цветов. np.unique возвращает список из отсортированных по уникальности элементов. Дальеш считем длину списка.
    
    #Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #берет минимальное и максимальные значения из 1 столбца масиива. : означает взять все строки, а 0 на индекс столбца.
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 #аналогично со вторым стобцом
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)) #Берет два одномерных списка чисел (координаты x и y) и превращает их в две двумерные таблицы. Создаем матрицу для метода predict из персептрона.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #np.arrange(start, stop, step, dtype). Формирует последовательность, можно указать начало, конеч, шаг и тип данных элемента массива.
    Z = Z.reshape(xx1.shape) #Меняет форму массива или матрицы в нужную нам
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) #Cтроит заполненные цветом контурные графики (изолинии с заливкой).
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    #Высести образы по классам
    for idx, cl in enumerate(np.unique(y)): # enumerate — встроенный инструмент, который позволяет в цикле одновременно получать и сам элемент коллекции (например, списка или строки), и его порядковый номер (индекс).
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left')
plt.show()
    
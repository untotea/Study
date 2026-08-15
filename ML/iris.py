import os
import pandas as pd
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data') #Берем набор данных Iris из хранилища машинного обучения UCI универа.
#print('URL:', s)
df = pd.read_csv(s, header=None, encoding='utf-8') #df - DataFrame. Загружаем данные сюда и читаем их. указываем кодировку.
df.tail()

import matplotlib.pyplot as plt
import numpy as np

# Выбрать ирис щетинистый и ирис разноцветный
y = df.iloc[0:100, 4].values #С помощью iloc берем первые 100 строк, где нам нужен только 4 столбец по индексу, выводим числовые значения.
y = np.where(y == 'Iris-setosa', -1, 1) # Проверка условия, если удовлетворяет, то 1, если нет, то -1

# Извлечь длину чашелистника и длину лепестка
X = df.iloc[0:100, [0, 2]].values #Извлекаем там же способом длины

# Вычертить график данных
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='щетинистый') #строит диаграмму с двумя осями
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный') #тоже самое, но с другим видом ириса
plt.xlabel('длина чашелистика [см]')
plt.ylabel('длина лепестка [см]')
plt.legend(loc='upper left') #добавляет лейблы в нужные нам место
plt.show()
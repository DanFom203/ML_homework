import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


# 1
bikes = pd.read_csv("bikes_rent.csv")


# 2
def plot_linear_regression(weathersit, cnt):
    model = LinearRegression()
    model.fit(weathersit.values.reshape(-1, 1), cnt)

    predictions = model.predict(weathersit.values.reshape(-1, 1))

    plt.scatter(weathersit, cnt, color='green')
    plt.plot(weathersit, predictions, color='red', label='линия регрессии')
    plt.xlabel('погодa')
    plt.ylabel('cпрос')
    plt.legend()
    plt.show()


cnt = bikes['cnt']
weathersit = bikes['weathersit']
plot_linear_regression(weathersit, cnt)


# 3
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    # Оценка точности
    accuracy = model.score(X, y)
    return accuracy


X = bikes[['temp', 'season']]
y = bikes['cnt']

accuracy = train_linear_regression(X, y)
print("Точность модели:", accuracy)


# 4
def plot_2d_prediction(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    model = LinearRegression()
    model.fit(X_pca, y)

    plt.scatter(X_pca[:, 0], y, label='Фактическое cnt')
    plt.scatter(X_pca[:, 0], model.predict(X_pca), color='red', label='Предсказанное cnt')
    plt.xlabel('Признак')
    plt.ylabel('cnt')
    plt.legend()
    plt.show()


X = bikes[['temp', 'season']]
y = bikes['cnt']
plot_2d_prediction(X, y)


# 5
def find_most_influential_feature(X, y, alpha=0.1):
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X, y)

    # Определение признака с наибольшим абсолютным коэффициентом
    index = abs(lasso_model.coef_).argmax()
    max_value = X.columns[index]

    return max_value


X = bikes.drop(columns=['cnt'])
y = bikes['cnt']
most_value = find_most_influential_feature(X, y)
print("Признак с наибольшим влиянием:", most_value)
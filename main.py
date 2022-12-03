import warnings
import numpy as np
from math import *
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from plotly.offline import init_notebook_mode, plot
from plotly import graph_objs as go
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

init_notebook_mode(connected=True)
warnings.simplefilter("ignore")

def plotly_df(df, title = ''):
    data = []
    for column in df.columns:
        trace = go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column
        )
        data.append(trace)

    layout = dict(title=title)
    fig = dict(data=data, layout=layout)
    plot(fig, show_link=False)

#полученние данных(цены акции) из сsv файла
data_set = pd.read_csv('//Users//semensavcenko//Downloads//ll.csv')
msk = (data_set.index < len(data_set) - 35)
data_set_train = data_set[msk].copy()
data_set_test = data_set[~msk].copy()

#печать первых 5 значений
print(data_set_train.head())
#принт датасета
#plotly_df(data_set, 'price')

#проверка на пропуски, если сумма = 0, то пропусков нет
print(data_set_train.isnull().sum())
#удаляем если они есть
data_set.dropna()

#рисуем гистограмму
plt.hist(data_set_train, 20, facecolor='purple')
plt.show()

#рисуем коррелограмму с лагом от 1 до 60
tsaplots.plot_acf(data_set_train, lags=60, color='g')
plt.show()

#рисуем ЧАКФ без указания метода коэф будет выходить за границы интервала [-1;1]
plot_pacf(data_set, lags=60, method='ywmle', color='purple')
plt.show()

#проверим ряд на стационарность при помощи теста Дукки-Фуллера
#H0: Временной ряд является нестационарным.
print("Проверка на стационарность - \n")
test = sm.tsa.adfuller(data_set_train)
print("p-value: ", test[1])
print("Critical values: ", test[4]['5%'])

if test[1] > test[4]['5%']:
    print('H0 не отвергается')
else:
    print('H0 отверагется')

#проверка на наличие тренда критерием Кокса-Стюарта
#H0: тренд существует
def Foster_Stuart_test(X, p_level=0.95):
    a_level = 1 - p_level
    X = np.array(X)
    n = len(X)

    u = l = list()
    Xtemp = np.array(X[0])
    for i in range(1, n):
        Xmax = np.max(Xtemp)
        Xmin = np.min(Xtemp)
        u = np.append(u, 1 if X[i] > Xmax else 0)
        l = np.append(l, 1 if X[i] < Xmin else 0)
        Xtemp = np.append(Xtemp, X[i])

    d = np.int64(np.sum(u - l))

    mean_d = 0
    mean_S = 2 * np.sum([1 / i for i in range(2, n + 1)])
    std_d = sqrt(mean_S)

    t_d = (d - mean_d) / std_d

    # табличные значения статистики критерия
    df = n
    t_table = sci.stats.t.ppf((1 + p_level) / 2, df)

    # проверка гипотезы
    conclusion_d = 'H0 отвергается' if t_d <= t_table else 'H0 не отвергается'

    # формируем результат
    result = pd.DataFrame({
        'n': (n),
        'p_level': (p_level),
        'a_level': (a_level),
        'coef:': (t_d),
        'crit_value': (t_table),
        'coef ≤ crit_value': (t_d <= t_table),
        'conclusion': conclusion_d
    },
        index=['Foster_Stuart_test'])

    return result

result = Foster_Stuart_test(data_set_train)
pd.set_option('display.max_columns', None)
print("Проверка на тенеденцию - \n")
display(result)

#прогнозирование на основе модели ARIMA
#AR - auto-regressive(p) означает сколько параметров
#будут включены в модель при прогнозировании
#I - Integrated помогает нам сделать ряд стационарным
#MA - Moving Average означает что временной ряд может быть
#регрессирован на прошлых ошибках
print("\n")
print("\t\t\t\t\t\t\t\tARIMA")
model = ARIMA(data_set_train, order=(1, 1, 1)).fit()
print(model.summary())
#оценка значимости модели

#адекватность модели
myresiduals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(1, 2)
myresiduals.plot(title="Residuals", ax=ax[0])
myresiduals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#прогноз
forecast = model.forecast(35)
data_set['forecast'] = [None] * len(data_set_train) + list(forecast)
data_set.plot()
plt.show()
#оценка качества модели
#среднеквадратичное отклонение
mse = np.sqrt(mean_squared_error(data_set_test, forecast))
print('mse: ', mse)

#средняя абсолютная ошибка
mae = mean_absolute_error(data_set_test, forecast)
print('mae: ', mae)

#средняя абсолютная ошибка в %
mape = mean_absolute_percentage_error(data_set_test, forecast)
print(f'mape: {mape}%')
#по моим данных следующие значения
#mse:  1636.8611498373427
#mae:  1404.5483420297503
#mape: 0.30054720137066887%
#Величина MAPE подсчитанная программой получилась 0,3 %.
#Имея данные, не использованные в прогнозе,
#мы получили MAPE, равную 0,3%.
#Значит, точность прогноза составляет – 99,7%.
#Такая точность прогноза является хорошей


#прогнозирование на основе модели

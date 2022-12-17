import warnings
import numpy as np
from math import *
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.graphics import tsaplots
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from plotly.offline import init_notebook_mode, plot
from plotly import graph_objs as go
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

init_notebook_mode(connected=True)
warnings.simplefilter("ignore")


def plotly_df(df, title=''):
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


i = 48
#полученние данных(цены акции) из сsv файла
data_set = pd.read_csv('//Users//semensavcenko//Downloads//ll.csv')
msk = (data_set.index < len(data_set) - i)
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
#plt.hist(data_set_train, 20, facecolor='purple')
#plt.show()

#рисуем коррелограмму с лагом от 1 до 60
tsaplots.plot_acf(data_set_train, lags=20, color='g')
plt.show()

#рисуем ЧАКФ без указания метода коэф будет выходить за границы интервала [-1;1]
plot_pacf(data_set, lags=20, method='ywmle', color='purple')
plt.show()

#проверим ряд на стационарность при помощи теста Дукки-Фуллера
#H0: Временной ряд является нестационарным.
print("Проверка на стационарность - \n")
test = sm.tsa.adfuller(data_set_train)
print ('adf: ', test[0])
print("p-value: ", test[1])
print("Critical values: ", test[4]['5%'])

if test[0] > test[4]['5%']:
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
otg1diff = data_set_train.diff(periods=1).dropna()

test1 = sm.tsa.adfuller(otg1diff)
print("adf: ", test1[0])
print("p-value: ", test1[1])
print("Critical values: ", test1[4]['5%'])

if test1[0] > test1[4]['5%']:
    print('H0 не отвергается')
else:
    print('H0 отверагется')
plot_pacf(otg1diff, lags=20, title='PACF for AR')
tsaplots.plot_acf(otg1diff, lags=20, title='ACF for MA')

print("\n")
print("\t\t\t\t\t\t\t\tARIMA")
model = ARIMA(data_set_train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend='t').fit()
print(model.summary())
#оценка значимости модели

#адекватность модели
myresiduals = pd.DataFrame(model.resid)
_, ax = plt.subplots(1, 2)
myresiduals.plot(title="Residuals", ax=ax[0])
myresiduals.plot(kind='kde', title='Density', ax=ax[1])


_, axes = plt.subplots(2, 2, sharex=True)
axes[0, 0].plot(data_set_train)
axes[0, 0].set_title('The Genuine Series')
tsaplots.plot_acf(data_set_train, ax=axes[0, 1])
# Order of Differencing: First
axes[1, 0].plot(data_set_train.diff())
axes[1, 0].set_title('Order of Differencing: First')
tsaplots.plot_acf(data_set_train.diff().dropna(), ax=axes[1, 1])
#прогноз
forecast = model.forecast(i)
print(len(data_set_train))
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
print(f'mape: {mape*100}%')
print("END ARIMA")


model_auto_arima = auto_arima(data_set_train, start_p=0, d=1, start_q=0)
print(model_auto_arima)
pred = model_auto_arima.predict(n_periods=i)
plt.plot(data_set_test)
plt.plot(pred)
plt.show()
mape = mean_absolute_percentage_error(data_set_test, pred)
print(f'mape: {mape*100}%')
#для отчета
#print(data_set_test.tail(35))


#прогнозирование на основе модели Двойного exp сглаживания, т.к. он учитывает тендецию, в отличие от модели 1 порядка
model2 = ExponentialSmoothing(np.asarray(data_set_train), trend='add', seasonal=None, damped_trend=True).fit()
forecast2 = model2.forecast(i)
print(model2.params)
data_set['forecast'] = ''
data_set['forecast2'] = [None] * len(data_set_train) + list(forecast2)
data_set.plot()
plt.show()
#адекватность модели
myres = pd.DataFrame(model2.resid)
_, ax = plt.subplots(1, 2)
myres.plot(title="Residuals", ax=ax[0])
myres.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

#оценка качества модели
#среднеквадратичное отклонение
mse = np.sqrt(mean_squared_error(data_set_test, forecast2))
print('mse: ', mse)

#средняя абсолютная ошибка
mae = mean_absolute_error(data_set_test, forecast2)
print('mae: ', mae)

#средняя абсолютная ошибка в %
mape = mean_absolute_percentage_error(data_set_test, forecast2)
print(f'mape: {mape * 100}%')
print("END LIN EXP")

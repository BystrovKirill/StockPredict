from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, logout, authenticate
from django.utils import timezone
from django.contrib.auth.decorators import login_required

from .models import Company

# Финансовые данные
import yfinance as yf

# Обучение модели
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm
import numpy as np
import pandas as pd

# Построение графиков
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Scatter
import datetime as dt

def home(request):
    return render(request, 'stock/home.html')


def signupuser(request):
    if request.method == 'GET':
        return render(request, 'stock/signupuser.html', {'form': UserCreationForm()})
    else:
        if request.POST['password1'] == request.POST['password2']:
            try:
                user = User.objects.create_user(request.POST['username'], password=request.POST['password1'])
                user.save()
                login(request, user)
                return redirect('home')
            except IntegrityError:
                return render(request, 'stock/signupuser.html',
                              {'form': UserCreationForm(),
                               'error': "Имя пользователя занято. Пожалуйста выберите другое."})
        else:
            return render(request, 'stock/signupuser.html',
                          {'form': UserCreationForm(), 'error': "Пароли не совпадают."})


def loginuser(request):
    if request.method == 'GET':
        return render(request, 'stock/loginuser.html', {'form': AuthenticationForm()})
    else:
        user = authenticate(request, username=request.POST['username'], password=request.POST['password'])
        if user is None:
            return render(request, 'stock/loginuser.html',
                          {'form': AuthenticationForm(), 'error': 'Имя пользователя и/или пароль введены неправильно.'})
        else:
            login(request, user)
            return redirect('home')


@login_required
def logoutuser(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')


@login_required
def companies(request):
    companies = Company.objects.all()
    return render(request, 'stock/companies.html', {'companies': companies})


def about(request):
    return render(request, 'stock/about.html')


# def predict(request, slug, number_of_days):
#    return render(request, 'stock/predict.html', {'slug': slug, 'number_of_days': number_of_days})


@login_required
def predict(request, slug):
    # Построение графика актуальных цен акций конекретной компании
    df = yf.download(tickers=slug, period='1d', interval='1m')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'], name='market data'))
    fig.update_layout(
        title='{} live share price evolution'.format(slug),
        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')

    # Обучение модели линейной регрессии
    try:
        df_ml = yf.download(tickers=slug, period='6mo', interval='1h')
    except:
        df_ml = yf.download(tickers='AAPL', period='6mo', interval='1h')

    df_ml = df_ml[['Adj Close']]
    number_of_days = 7
    forecast_out = number_of_days
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # Splitting data for Test and Train
    X = np.array(df_ml.drop(['Prediction'], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df_ml['Prediction'])
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    # Применяем Линейную регрессию
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    # Предсказываем R2
    confidence = clf.score(X_test, y_test)
    # Предсказываем для 'н-ого' количества дней
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()

    # Строим график для предсказанного значения
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])

    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('stock/Data/Tickers.csv')
    to_search = slug
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                      'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0, ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================

    return render(request, 'stock/predict.html', context={'slug': slug,
                                                   'plot_div': plot_div,
                                                   'confidence': confidence,
                                                   'forecast': forecast,
                                                   'ticker_value': slug,
                                                   'number_of_days': number_of_days,
                                                   'plot_div_pred': plot_div_pred,
                                                   'Symbol': Symbol,
                                                   'Name': Name,
                                                   'Last_Sale': Last_Sale,
                                                   'Net_Change': Net_Change,
                                                   'Percent_Change': Percent_Change,
                                                   'Market_Cap': Market_Cap,
                                                   'Country': Country,
                                                   'IPO_Year': IPO_Year,
                                                   'Volume': Volume,
                                                   'Sector': Sector,
                                                   'Industry': Industry,
                                                   })

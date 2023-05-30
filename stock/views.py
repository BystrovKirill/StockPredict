from django.core.paginator import Paginator
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, logout, authenticate
from django.urls import reverse_lazy
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.views.generic import FormView

from .models import Company, Contact
from . import utils

# Финансовые данные
import requests
import apimoex

# Обучение модели
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import LSTM, Dense
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
    paginator = Paginator(companies, 3)

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'stock/companies.html', {'page_obj': page_obj, 'companies': companies})


def about(request):
    if request.method == 'POST':
        contact = Contact()
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        contact.name = name
        contact.email = email
        contact.message = message
        contact.save()
        return redirect('home')
    return render(request, 'stock/about.html')


@login_required
def search_companies(request):
    if request.method == 'POST':
        searched = request.POST['searched'].title()
        stocks = Company.objects.filter(Q(name__icontains=searched)|Q(slug__icontains=searched))
        return render(request, 'stock/searchcompanies.html', {'searched': searched, 'stocks': stocks})
    else:
        return render(request, 'stock/searchcompanies.html', {})


@login_required
def predict(request, slug):
    # получаем название компании по слагу
    name = utils.name_dict_ru_reverse[slug]
    company = Company.objects.get(slug=slug)

    # загрузка данных компании по ее слагу
    with requests.Session() as session:
        data = apimoex.get_board_history(session, slug)
        df = pd.DataFrame(data)
        df.set_index('TRADEDATE', inplace=True)

    # Построение графика актуальных цен акций конекретной компании
    fig = go.Figure()
    fig = go.Figure([go.Scatter(x=df.index, y=df['CLOSE'])])
    fig.update_layout(
        title='Временная шкала',
        yaxis_title='Стоимость акции (Руб.)')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=45, label="day", step="day", stepmode="todate"),
                dict(count=1, label="month", step="month", stepmode="backward"),
                dict(count=3, label="year", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')

    # Обучение модели линейной регрессии
    df = df[['CLOSE']]
    number_of_days = 10
    forecast_out = number_of_days
    df['Prediction'] = df[['CLOSE']].shift(-forecast_out)

    # Разбиваем входные данные на обучающую и проверочную выборки
    X = np.array(df.drop(['Prediction'], axis=1))
    X = np.nan_to_num(X)
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(df['Prediction'])
    y = np.nan_to_num(y)
    y = y[:-forecast_out]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Применяем Линейную регрессию
    lin_regr = LinearRegression()
    lin_regr.fit(X_train, y_train)

    # Предсказываем R2
    confidence = lin_regr.score(X_test, y_test)

    # Предсказываем для 'н-ого' количества дней
    forecast_prediction = lin_regr.predict(X_forecast)
    forecast = forecast_prediction.tolist()

    # Строим график для предсказанного значения линейной регрессии
    pred_dict = {"Date": [], "Prediction": []}
    for i in range(0, len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])

    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(
        title='Временная шкала',
        yaxis_title='Стоимость акции (Руб.)')
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Параметры для рендера страницы ========================================
    return render(request, 'stock/predict.html', context={'name': name,
                                                          'company': company,
                                                          'slug': slug,
                                                          'plot_div': plot_div,
                                                          'confidence': confidence,
                                                          'forecast': forecast,
                                                          'ticker_value': slug,
                                                          'number_of_days': number_of_days,
                                                          'plot_div_pred': plot_div_pred,
                                                          })

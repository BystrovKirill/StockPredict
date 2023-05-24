from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, logout, authenticate
from django.utils import timezone
from django.contrib.auth.decorators import login_required

from .models import Company

# Построение графиков и обучение модели
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from plotly.graph_objs import Scatter


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
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="я")
    plot_div = plot(fig, auto_open=False, output_type='div')

    return render(request, 'stock/predict.html', context={'slug': slug,
                                                          'plot_div': plot_div,
                                                          })

from django.contrib import admin
from django.urls import path

from stock import views


urlpatterns = [
    path('admin/', admin.site.urls),

    # Auth
    path('signup/', views.signupuser, name='signupuser'),
    path('login/', views.loginuser, name='loginuser'),
    path('logout/', views.logoutuser, name='logoutuser'),

    # Stock
    path('', views.home, name='home'),
    path('companies/', views.companies, name='companies'),
    #path('predict/<str:slug>/<str:number_of_days>/', views.predict, name='predict'),
    path('predict/<str:slug>/', views.predict, name='predict'),
    path('about/', views.about, name='about'),
]

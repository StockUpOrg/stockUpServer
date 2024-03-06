

from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.user_list, name='user-list'),
    path('users/create/', views.user_create, name='user-create'),
    path('users/<str:pk>/', views.user_detail, name='user-detail'),
    path('stocks/<str:identifier>/', views.stock_detail, name='stock_detail'),
    path('stock-symbol/<str:name>/', views.get_symbol, name='stock_symbol'),
    path('stock-news/<str:identifier>/', views.stock_news, name='stock_news'),
    path('stock-info/<str:identifier>/', views.stock_info, name='stock_info'),
    path('pre/<str:identifier>/', views.LinearRegModel, name='stock_predictions'),
]

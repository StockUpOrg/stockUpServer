

from django.urls import path
from . import views

urlpatterns = [
    path('users/', views.user_list, name='user-list'),
    path('users/create/', views.user_create, name='user-create'),
    path('users/<int:pk>/', views.user_detail, name='user-detail'),
    path('stocks/<str:identifier>/', views.stock_detail, name='stock_detail'),
]

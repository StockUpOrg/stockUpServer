from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import UserAccount
from .serializers import UserAccountSerializer
from django.http import JsonResponse, Http404
from .models import Stock
import yfinance as yf 
import requests
from django.conf import settings 
from datetime import datetime, timedelta
# from .ML_Function.LinearRegressionModel import LinearRegressionModel
from .ML_Function.LinearRegression import LinearRegressionModel
import numpy as np
from pandas_datareader import data as pdr

# For Machine Learning Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

@api_view(['GET'])
def user_list(request):
    if request.method == 'GET':
        users = UserAccount.objects.all()
        serializer = UserAccountSerializer(users, many=True)
        return Response(serializer.data)

@api_view(['POST'])
def user_create(request):
    if request.method == 'POST':
        serializer = UserAccountSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
def user_detail(request, pk):
    try:
        user = UserAccount.objects.get(user_id=pk)
    except UserAccount.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = UserAccountSerializer(user)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = UserAccountSerializer(user, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

def get_symbol_from_name(name):
    """
    Queries the Alpha Vantage API to get the stock symbol for a given company name.
    Returns the symbol if found, otherwise returns None.
    """
    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={name}&apikey={settings.ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    # Attempt to extract the best match if the search results are not empty
    matches = data.get('bestMatches', [])
    if matches:
        # Assuming the best match is the first one
        return matches[0]['1. symbol'], matches[0]['2. name']
    return None, None

def get_symbol(request, name):
    symbol, company_name = get_symbol_from_name(name)
    if symbol == None:
        return JsonResponse({'error': f"No symbol found for '{name}'."}, status=404)
    else:
        return JsonResponse(symbol,  safe=False)
    

def stock_detail(request, identifier):
    # First, try to resolve the identifier as a company name to a symbol
    symbol, company_name = get_symbol_from_name(identifier)
    if symbol == None:
        # If no symbol is found, assume the identifier might already be a symbol
        symbol = identifier.upper()
        company_name = identifier  # Fallback to the identifier if the name resolution fails
        

    # Fetch stock information using yfinance
    time = request.GET.get('time')
    if time :
        time_period = f"{time}y"
    else:
        time_period = "1y"    
    stock = yf.Ticker(symbol)
    hist = stock.history(period=time_period)

    if hist.empty:
        return JsonResponse({'error': f"No historical data found for '{identifier}'."}, status=404)

    stock_data = [{
        'date': date.strftime('%Y-%m-%d'),
        'open': row['Open'],
        'close': row['Close']
    } for date, row in hist.iterrows()][::-1]  # List is reversed to show latest data first

    response_data = {
        'symbol': symbol,
        'company_name': company_name,
        'stock_data': stock_data,
    }

    return JsonResponse(response_data)

def LinearRegModel(request, identifier):
    end = datetime.now()
    futureDays = 30
    days = request.GET.get('days')
    if days:
        futureDays = int(days)
    predictions = LinearRegressionModel(identifier,'Close', end, futureDays)
    future_dates_with_data = [(end + timedelta(days=i+1), predictions[i]) for i in range(len(predictions))]
    return JsonResponse({'result': future_dates_with_data})

def stock_news(request, identifier):
    symbol, company_name = get_symbol_from_name(identifier)
    if symbol == None:
        # If no symbol is found, assume the identifier might already be a symbol
        symbol = identifier.upper()
        company_name = identifier  # Fallback to the identifier if the name resolution fails

    stock = yf.Ticker(symbol)
    return JsonResponse({'stock_news': stock.news})

def stock_info(request, identifier):
    symbol, company_name = get_symbol_from_name(identifier)
    if symbol == None:
        # If no symbol is found, assume the identifier might already be a symbol
        symbol = identifier.upper()
        company_name = identifier  # Fallback to the identifier if the name resolution fails

    stock = yf.Ticker(symbol)
    return JsonResponse({'stock_info': stock.info})






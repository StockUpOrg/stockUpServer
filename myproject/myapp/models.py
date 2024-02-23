

from django.db import models

class UserAccount(models.Model):
    firstName = models.CharField(max_length=100)
    lastName = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    favorite_stocks = models.JSONField(default=list)

class Stock(models.Model):
    name = models.CharField(max_length=255, unique=True)
    symbol = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return f"{self.name} ({self.symbol})"
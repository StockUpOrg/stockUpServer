

from django.db import models
import uuid

class UserAccount(models.Model):
    user_id = models.UUIDField( default=uuid.uuid4)
    firstName = models.CharField(max_length=100)
    lastName = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    favorite_stocks = models.JSONField(default=list)

class Stock(models.Model):
    name = models.CharField(max_length=255, unique=True)
    symbol = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return f"{self.name} ({self.symbol})"
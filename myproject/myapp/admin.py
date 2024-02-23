from django.contrib import admin
from .models import UserAccount
from .models import Stock

admin.site.register(UserAccount)
admin.site.register(Stock)

# Generated by Django 5.0.2 on 2024-02-22 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_stock'),
    ]

    operations = [
        migrations.AddField(
            model_name='stock',
            name='last_close',
            field=models.DecimalField(decimal_places=2, max_digits=10, null=True),
        ),
    ]

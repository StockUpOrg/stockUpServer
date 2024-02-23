# Generated by Django 5.0.2 on 2024-02-23 22:58

import uuid
from django.db import migrations, models

def create_uuid(apps, schema_editor):
    UserAccount = apps.get_model('myapp', 'UserAccount')
    for user in UserAccount.objects.all():
        user.user_id = uuid.uuid4()
        user.save()

class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0004_remove_stock_last_close_alter_stock_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='useraccount',
            name='user_id',
            field=models.UUIDField(default=uuid.uuid4),
        ),
        migrations.RunPython(create_uuid),
        migrations.AlterField(
            model_name='useraccount',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]

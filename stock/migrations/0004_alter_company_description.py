# Generated by Django 4.2.1 on 2023-05-28 13:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0003_remove_company_linear_regr_model'),
    ]

    operations = [
        migrations.AlterField(
            model_name='company',
            name='description',
            field=models.TextField(max_length=3000, verbose_name='Описание'),
        ),
    ]
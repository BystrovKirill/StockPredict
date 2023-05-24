# Generated by Django 4.2.1 on 2023-05-23 12:55

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Company',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50, unique=True, verbose_name='Название компании')),
                ('description', models.TextField(max_length=500, verbose_name='Описание')),
                ('slug', models.CharField(blank=True, max_length=50, unique=True, verbose_name='Название котировки')),
            ],
            options={
                'verbose_name': 'Название компании',
                'verbose_name_plural': 'Название компаний',
            },
        ),
    ]
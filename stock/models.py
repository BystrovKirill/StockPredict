from django.db import models

from captcha.fields import CaptchaField

from . import utils


class Company(models.Model):
    name = models.CharField(max_length=50, verbose_name='Название компании', unique=True)
    description = models.TextField(blank=False, max_length=3000, verbose_name='Описание')
    slug = models.CharField(max_length=50, blank=True, unique=True, verbose_name='Название котировки')
    photo = models.ImageField(upload_to="photo", blank=True)

    class Meta:
        verbose_name = 'Название компании'
        verbose_name_plural = 'Название компаний'
        ordering = ['name']

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = utils.name_dict_ru.get(self.name.lower(), 'Такой компании нет в списке')
        super().save(*args, **kwargs)


class Contact(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField(max_length=50)
    message = models.CharField(max_length=500)
from django.db import models

from . import utils


class Company(models.Model):
    name = models.CharField(max_length=50, verbose_name='Название компании', unique=True)
    description = models.TextField(blank=False, max_length=500, verbose_name='Описание')
    slug = models.CharField(max_length=50, blank=True, unique=True, verbose_name='Название котировки')

    class Meta:
        verbose_name = 'Название компании'
        verbose_name_plural = 'Название компаний'

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = utils.name_dict.get(self.name, 'Такой компании нет в списке')
        super().save(*args, **kwargs)

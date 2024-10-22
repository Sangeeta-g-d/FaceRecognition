from django.db import models
from django.utils import timezone


# Create your models here.


class People(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)

    def __str__(self):
        return self.name


class Attendance(models.Model):
    person = models.ForeignKey(People, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    time = models.TimeField(default=timezone.now)

    def __str__(self):
        return f"Attendance for {self.user.name} on {self.date}"
# recognition/forms.py
from django import forms

class PeopleForm(forms.Form):
    id = forms.IntegerField(label='Student ID')
    name = forms.CharField(label='Student Name', max_length=100)
    age = forms.IntegerField(label='Student Age')
    gender = forms.CharField(label='Student Gender', max_length=10)


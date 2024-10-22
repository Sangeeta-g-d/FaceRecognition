# recognition/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('admin_login',views.admin_login,name="admin_login"),
    path('admin_db',views.admin_db,name="admin_db"),
    path('register_students/', views.register_students, name='register_students'),
    path('success/', views.success, name='success'),
    path('train/', views.train, name='train'),
    path('detect/', views.detect, name='detect'),
]

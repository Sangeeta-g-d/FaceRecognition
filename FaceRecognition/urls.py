"""
URL configuration for FaceRecognition project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.contrib import admin
from django.urls import path
from recognition import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('admin_login/', views.admin_login, name='admin_login'),
    path('admin_db/', views.admin_db, name='admin_db'),
    path('view_students/', views.view_students, name='view_students'),
    path('register_students/', views.register_students, name='register_students'),
    path('attendance_list/', views.attendance_list, name='attendance_list'),
    path('train/', views.train, name='train'),
    path('detect/', views.detect, name='detect'),
    path('admin_logout', views.admin_logout, name='admin_logout'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
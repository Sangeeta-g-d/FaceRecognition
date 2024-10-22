# recognition/views.py
import os
import cv2
import numpy as np
import sqlite3
from PIL import Image
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponse
from .forms import PeopleForm
from .models import People, Attendance
from django.utils import timezone
import time
from datetime import datetime, date
from django.utils.dateparse import parse_date


def index(request):
    return render(request, 'index.html')


def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None and user.is_superuser:
            login(request, user)
            return redirect('/admin_db')
        else:
            error_message = "Invalid username or password"
            return render(request, 'admin_login.html', {'error_message': error_message})
    return render(request, 'admin_login.html')

def admin_db(request):
    return render(request, 'admin_db.html')


from django import forms
from django.shortcuts import render, redirect
from django.core.exceptions import ValidationError
from .models import People

class PeopleForm(forms.Form):
    id = forms.IntegerField(label='Student ID')
    name = forms.CharField(label='Student Name', max_length=100)
    age = forms.IntegerField(label='Student Age')
    gender = forms.CharField(label='Student Gender', max_length=10)

    def clean_id(self):
        id = self.cleaned_data.get('id')
        if People.objects.filter(id=id).exists():
            raise ValidationError('Student with this ID already exists.')
        return id

def register_students(request):
    error_message = None
    if request.method == 'POST':
        form = PeopleForm(request.POST)
        if form.is_valid():
            Id = form.cleaned_data['id']
            Name = form.cleaned_data['name']
            Age = form.cleaned_data['age']
            Gen = form.cleaned_data['gender']

            # Assuming these functions are defined elsewhere in your code
            insert_or_update(Id, Name, Age, Gen)
            capture_face_images(Id)
            return redirect('train')
        else:
            error_message = form.errors.get('id', [None])[0]
    else:
        form = PeopleForm()
    return render(request, 'register_students.html', {'form': form, 'error_message': error_message})


def view_students(request):
    search_query = request.GET.get('search', '')
    selected_id = request.GET.get('id', '')

    data = People.objects.all()

    if search_query:
        data = data.filter(name__icontains=search_query)

    if selected_id and selected_id != 'all':
        data = data.filter(id=selected_id)

    unique_ids = People.objects.values_list('id', flat=True).distinct()

    context = {
        'data': data,
        'search_query': search_query,
        'selected_id': selected_id,
        'unique_ids': unique_ids,
    }

    return render(request, 'view_students.html', context)



def attendance_list(request):
    attendances = Attendance.objects.select_related('person').all().order_by('-date')

    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    if start_date:
        attendances = attendances.filter(date__gte=parse_date(start_date))
    if end_date:
        attendances = attendances.filter(date__lte=parse_date(end_date))

    context = {
        'attendances': attendances,
        'start_date': start_date,
        'end_date': end_date,
    }

    return render(request, 'attendance_list.html', context)



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAARCASCADE_PATH = os.path.join(BASE_DIR, 'recognition/haarcascade/haarcascade_frontalface_default.xml')
DATASET_PATH = os.path.join(BASE_DIR, 'recognition/dataset')
TRAINER_PATH = os.path.join(BASE_DIR, 'recognition/recognizer/trainingdata.yml')
DB_PATH = os.path.join(BASE_DIR, 'db.sqlite3')

faceDetect = cv2.CascadeClassifier(HAARCASCADE_PATH)

def insert_or_update(Id, Name, Age, Gen):
    people, created = People.objects.update_or_create(
        id=Id,
        defaults={'name': Name, 'age': Age, 'gender': Gen},
    )

def capture_face_images(Id):
    cam = cv2.VideoCapture(0)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"recognition/dataset/User.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(400)

        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sampleNum > 50:
            break

    cam.release()
    cv2.destroyAllWindows()

def get_images_with_ids(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in image_paths:
        faceImg = Image.open(single_image_path).convert("L")
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(ids), faces

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    ids, faces = get_images_with_ids(DATASET_PATH)
    recognizer.train(faces, ids)
    recognizer.save(TRAINER_PATH)
    cv2.destroyAllWindows()

def get_profile(id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT * FROM recognition_people WHERE id=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile


def save_attendance(profile):
    person = People.objects.get(id=profile[0])
    today = date.today()
    attendance = Attendance.objects.filter(person=person, date=today)
    
    if attendance.exists():
        return False
    else:
        now = timezone.now()
        Attendance.objects.create(person=person, date=today, time=now.time())
        return True

def detect_faces():
    cam = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    confidence_threshold = 50

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < confidence_threshold:
                profile = get_profile(id)
                if profile:
                    cv2.putText(img, f"Name: {profile[1]}", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)
                    cv2.imshow("Face", img)
                    cv2.waitKey(1000)  # Wait for 500 milliseconds
                    if save_attendance(profile):
                        cv2.putText(img, f"Attendance Marked for {profile[1]}", (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, f"Attendance marked for {profile[1]}", (x, y+h+45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Not registered..!", (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def register(request):
    if request.method == 'POST':
        form = PeopleForm(request.POST)
        if form.is_valid():
            Id = form.cleaned_data['id']
            Name = form.cleaned_data['name']
            Age = form.cleaned_data['age']
            Gen = form.cleaned_data['gender']

            insert_or_update(Id, Name, Age, Gen)
            capture_face_images(Id)
            return redirect('train')
    else:
        form = PeopleForm()
    return render(request, 'register.html', {'form': form})

def success(request):
    return render(request, 'success.html')

def train(request):
    if request.method == 'POST':
        train_model()
        return redirect('detect')
    else:
        return render(request, 'train.html')


def detect(request):
    if request.method == 'POST':
        detect_faces()
        return HttpResponse('Face detection Stopped.')
    return render(request, 'detect.html')


def admin_logout(request):
    logout(request)
    return redirect('/')
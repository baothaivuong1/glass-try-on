import requests

from .process_frame import process
import cv2

from django.shortcuts import render
from django.template.loader import get_template

from django.http import StreamingHttpResponse
from django.http import HttpResponse
from django.template.response import TemplateResponse

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
# Create your views here.


def index(request):
    return render(request, "webcam/index.html")

def stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break
        image = cv2.flip(image, 1)

        image = process(image)
        cv2.imwrite('demo.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
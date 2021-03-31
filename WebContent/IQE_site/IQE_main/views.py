from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
from pathlib import Path
from services.test_model import predict

# Imaginary function to handle an uploaded file.
# from services.test_model import predict

def landing(request) :
    BASE_DIR = Path(__file__).resolve().parent.parent
    context = {}
    if request.method == "POST" :
        uploaded_file = request.FILES["input_image"]
        fs = FileSystemStorage()
        name = fs.save("input.jpg", uploaded_file)
        image_path = os.path.join(BASE_DIR, "media/input.jpg")
        models_path = os.path.join(BASE_DIR, "services/models/")
        save_path = os.path.join(BASE_DIR, "static/images/output.jpg")
        predict(image_path, save_path, models_path)
        print("File Saved")
        fs.delete(name)
        return redirect('download/')
    return render(request, "index.html", context)
  
def download(response):
    BASE_DIR = Path(__file__).resolve().parent.parent
    context = {}
    context['s_path'] = os.path.join(BASE_DIR, "static/images/output.jpg")
    return render(response, "download.html", context)
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.template.loader import get_template
import datetime

from xhtml2pdf import pisa

from keras.models import load_model
import tensorflow as tf
from tensorflow import Graph
import json
import numpy as np

img_height, img_width = 256, 256
with open('./models/labels.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/VGG16_256x256_yes_oversamp_epo_100_bs_32.h5')


def index(request):
    if request.method =='POST':
        # saving the image 
        fileObj = request.FILES['filePath']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathNamePDF = fs.path(filePathName)
        # contains the path of the image for viewing in index.html
        filePathName = fs.url(filePathName)
        request.session['filePathName'] = filePathName
        # get path for PDF 
        request.session['filePathNamePDF'] = filePathNamePDF
        return redirect(predictImage)
    
    return render(request, 'index.html')


def predictImage(request):
    # prediction  
    filePathName = request.session.get('filePathName')
    testimage='.'+filePathName
    img = tf.keras.utils.load_img(testimage, target_size=(img_height, img_width))
    x = tf.keras.utils.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)


    predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    
    # store predicted label in session for PDF 
    request.session['predictedLabel'] = predictedLabel
    
    context={'filePathName':filePathName, 'predictedLabel':predictedLabel}
    return render(request,'prediction.html', context)


def renderPDF(request, *args, **kwargs):
    template_path = 'pdf.html'

    filePathNamePDF = request.session.get('filePathNamePDF')
    predictedLabel = request.session.get('predictedLabel')
    dateToday = datetime.date.today()
    timeToday = datetime.datetime.now().time()

    fileName = filePathNamePDF.split("media\\")

    context = {'title': 'Anterior Cruciate Ligament (ACL) Injury Classification System Results',
               'filePathNamePDF' : filePathNamePDF,
               'predictedLabel' : predictedLabel,
               'dateToday' : dateToday,
               'timeToday' : timeToday,
               'fileName' : fileName[1]}

    response = HttpResponse(content_type='application/pdf')

    # if display
    response['Content-Disposition'] = 'filename="ACL Injury Prediction.pdf"'
    template = get_template(template_path)
    html = template.render(context)

    # create pdf
    pisa_status = pisa.CreatePDF(html, dest=response)
    # if error, show some view
    if pisa_status.err:
        return HttpResponse('Error <pre>' + html + '</pre>')
    return response
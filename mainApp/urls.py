from django.urls import path

from . import views

urlpatterns = [
    #Leave as empty string for base url
	path('', views.index, name="homepage"),
    path('predictImage',views.predictImage,name='predictImage'),
    path('renderPDF', views.renderPDF, name='renderPDF')
]

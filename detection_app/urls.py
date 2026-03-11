from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('results/<uuid:pk>/', views.results_view, name='results'),
    path('error/', views.error_view, name='error'),
]
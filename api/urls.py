from django.urls import path
from .views import ask_ai, health
from .jarvis_api import jarvis_audio, jarvis_organise

urlpatterns = [
    path('ask-ai/', ask_ai),
    path('health/', health),
    path('jarvis/organise/', jarvis_organise),
    path('jarvis/audio/', jarvis_audio),
]

from django.urls import path
from .views import ask_ai

urlpatterns = [
    path('ask-ai/', ask_ai),
]

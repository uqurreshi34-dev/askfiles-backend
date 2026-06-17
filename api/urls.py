from django.urls import path
from .views import ask_ai, health

urlpatterns = [
    path('ask-ai/', ask_ai),
    path('health/', health),
]

from django.urls import path
from .views import ask_ai, health, expand_query

urlpatterns = [
    path('ask-ai/', ask_ai),
    path('health/', health),
    path('expand-query/', expand_query),
]

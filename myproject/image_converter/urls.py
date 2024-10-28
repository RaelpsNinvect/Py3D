# image_converter/urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import index, upload_image
from .views import account_view
from .views import about_view

urlpatterns = [
    path('', index, name='index'),
    path('upload/', upload_image, name='upload_image'),
    path('account/', account_view , name='account'),
    path('about/', about_view , name='about'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
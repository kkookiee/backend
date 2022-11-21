from django.urls import path

from . import views
from django.conf import settings
from django.conf.urls.static import static

# user_image_upload에서 () 붙이면 self 오류남
urlpatterns = [
    path('', views.ListPost.as_view()),
    path('<int:pk>/', views.DetailPost.as_view()),
    path('upload/',views.model_form_upload,name='upload'),
    path('upload/image/',views.user_image_upload,name='upload_image'),
    path('upload/image/processing/',views.user_image_processing,name='image_processing')
]# + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


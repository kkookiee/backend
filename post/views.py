import json
import os

from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework import generics

from .forms import DocumentForm,ImageForm
from .models import Post,UserUploadImage
from .serializers import PostSerializer,ImagePostSerializer
from .processing.main import Main


class ListPost(generics.ListCreateAPIView):
    queryset = UserUploadImage.objects.all()
    serializer_class = ImagePostSerializer

class DetailPost(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    
###########################################################
# https://jakpentest.tistory.com/77

# text only
@method_decorator(csrf_exempt, name="dispatch")
def model_form_upload(request):
    print('model_form_upload')
    print(request.POST)
    if request.method == "POST":
        form = DocumentForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse(json.dumps({"status": "Success"}))
        else:
            return HttpResponse(json.dumps({"status": "Failed"}))
        
# image only        
@method_decorator(csrf_exempt, name="dispatch")
def user_image_upload(request):
    print('user_image_upload')
    if request.method == "POST":
        form = ImageForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponse(json.dumps({"status": "Success"}))
        else:
            return HttpResponse(json.dumps({"status": "Failed"}))        

# 이미지 처리 시작
@method_decorator(csrf_exempt, name="process")
def user_image_processing(request):
    print('user_image_processing')
    print(request.GET) # 쿼리 dict 여기엔 이미지 없음
    if request.method == "GET":
        path = request.GET.get('user_image')
        if len(path) != 0:
            sess = Main()
            # heatmap_url,proba,isForgery = sess.run(path)
            heatmap_url,proba,isForgery = sess.run(path)
            
            return HttpResponse(json.dumps({"status": "Success",
                                            "heatmap":heatmap_url,
                                            "result":{"proba":proba,
                                                    "isForgery":isForgery}
                                            }))
        else :
            return HttpResponse(json.dumps({"status": "Failed"}))
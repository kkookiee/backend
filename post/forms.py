from django import forms
from .models import Post,UserUploadImage # db

class DocumentForm(forms.ModelForm):
    class Meta :
        model = Post
        fields = ("title","content")
        
# front에서 Form을 구현할 때 이 내용 맞춰서 form 태그 구성하라
class ImageForm(forms.ModelForm):
    class Meta:
        model = UserUploadImage
        # fields = ('title','image')
        fields = ('image',)
        
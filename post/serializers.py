from rest_framework import serializers
from .models import Post,UserUploadImage

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            # 'id',
            'title',
            'content',
            # "image"
        )
        model = Post
        
class ImagePostSerializer(serializers.ModelSerializer):
    class Meta:
        fields = (
            # 'id' yet
            'image',
        )
        model = UserUploadImage
        
    
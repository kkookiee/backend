from django.contrib import admin
from .models import Post,UserUploadImage


# 원본
admin.site.register(Post)
admin.site.register(UserUploadImage)

# class PhotoInline(admin.TabularInline):
#     model = Photo

# Post 클래스는 해당하는 Photo 객체를 리스트로 관리하는 한다. 
# class PostAdmin(admin.ModelAdmin):
#     inlines = [PhotoInline, ]
  
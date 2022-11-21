from importlib.resources import contents
from django.db import models

# Post model : lab.js -> views.py
class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    
    def __str__(self):
        """molde title"""
        return self.title
    
class UserUploadImage(models.Model):
    # title = models.CharField(max_length=50)
    image = models.ImageField(upload_to='images/',blank=True, null=True)
    

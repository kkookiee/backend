# 테스트용 파일

from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
FRONT_DIR = BASE_DIR = Path(__file__).resolve().parent.parent.parent

a = os.path.join(FRONT_DIR,'frontend', 'src')
print(a)

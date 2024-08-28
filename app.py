from getImage import *
from main import *
import threading
import shutil
import os


def app():
    print("LUỒNG @ CHẠY")
    folderCache = "cache/"
    folderGlass = "glass/"
    folderNoGlass = "noglass/"
    for i in range(99999):
        imagePath, checksum = getImagex(folderCache)
        check = detectedGlass(image_path=imagePath)
        if check:
            shutil.copy(imagePath, folderGlass)
        else:
            shutil.copy(imagePath, folderNoGlass)
        os.remove(imagePath)


for i in range(10):
    threading.Thread(target=app).start()

from thispersondoesnotexist import get_online_person
from thispersondoesnotexist import *
from randoms import random_string


def getImagex(path):
    picture = get_online_person()
    checksum = get_checksum_from_picture(picture)
    imagePath = f'{path}{checksum}.jpeg'
    save_picture(picture, imagePath)
    return imagePath , checksum

[getImagex("image_list/") for _ in range(10000)]

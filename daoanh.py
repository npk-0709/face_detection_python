from PIL import Image 
def flip_resize(image,savex):
    img = Image.open(image)
    width, height = img.size
    new_width = int(width * 1.5)
    new_height = int(height * 1.5)
    img_reversed_vertical = img.transpose(Image.FLIP_LEFT_RIGHT)
    e = img_reversed_vertical.resize((new_width, new_height), Image.LANCZOS)
    e.save(savex)


flip_resize("image_2/243567.jpg","image2.jpg")
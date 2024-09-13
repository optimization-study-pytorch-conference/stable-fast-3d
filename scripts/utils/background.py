import os

import rembg

from sf3d.utils import remove_background, resize_foreground

def handle_image(image, rembg_session):
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, 0.85)
    image.save(os.path.join(os.getcwd(), "intermediate_image.png"))
    return image

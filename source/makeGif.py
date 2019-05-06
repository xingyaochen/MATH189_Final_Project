from PIL import Image, ImageDraw
import os


def makeGif(folder, duration, gifName):
    """
    Stitches all images in a directory into a GIF
    """
    if not os.path.exists('gifs'):
        os.mkdir('gifs')
    imgFiles = sorted(os.listdir(folder))
    images = [Image.open(folder+ "/" + fn) for fn in imgFiles if fn[0] != '.']
    images[0].save('gifs/'+gifName, format='GIF', append_images=images[1:], save_all=True, duration=duration, loop=0)


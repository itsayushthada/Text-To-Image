from zipfile import ZipFile
from pandas import read_csv
from PIL import Image
from io import BytesIO
import numpy as np
from pickle import dump, load
from cv2 import resize

###########################################################################
archive = ZipFile(file="flickr-image-dataset.zip", mode="r")

df = read_csv("results.csv", sep="|")
df = df.rename(columns=lambda x: x.strip())

IMAGELIST = df.loc[:, "image_name"].unique()

###########################################################################
images = {}
start_idx = 0
end_idx = 10000


for (idx, label) in enumerate(IMAGELIST[start_idx : end_idx]):

    img = archive.read("flickr30k_images/flickr30k_images/{}".format(label))
    img = BytesIO(img)
    img = np.asarray(Image.open(img))
    img = resize(img, (300,300))

    images[idx+start_idx] = img

result = np.array([images[key] for key in sorted(images.keys())])
with open('images1.pickle', 'wb') as file:
    dump(result, file)


del images
del file

###########################################################################
images = {}
start_idx = 10000
end_idx = 20000


for (idx, label) in enumerate(IMAGELIST[start_idx : end_idx]):

    img = archive.read("flickr30k_images/flickr30k_images/{}".format(label))
    img = BytesIO(img)
    img = np.asarray(Image.open(img))
    img = resize(img, (300,300))

    images[idx+start_idx] = img

result = np.array([images[key] for key in sorted(images.keys())])
with open('images2.pickle', 'wb') as file:
    dump(result, file)


del images
del file

###########################################################################
images = {}
start_idx = 10000
end_idx = None


for (idx, label) in enumerate(IMAGELIST[start_idx : ]):

    img = archive.read("flickr30k_images/flickr30k_images/{}".format(label))
    img = BytesIO(img)
    img = np.asarray(Image.open(img))
    img = resize(img, (300,300))

    images[idx+start_idx] = img

result = np.array([images[key] for key in sorted(images.keys())])
with open('images3.pickle', 'wb') as file:
    dump(result, file)




del file
del images
del result
del file

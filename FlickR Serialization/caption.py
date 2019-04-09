from zipfile import ZipFile
from pandas import read_csv
from io import BytesIO
import numpy as np
from pickle import dump

archive = ZipFile(file="flickr-image-dataset.zip", mode="r")

df = read_csv("results.csv", sep="|")
df = df.rename(columns=lambda x: x.strip())

IMAGELIST = df.loc[:, "image_name"].unique()

captions = {}

for (idx, label) in enumerate(IMAGELIST):
    captions[idx] = list(df.loc[df["image_name"] == label, "comment"].values)
    
with open('captions.pickle', 'wb') as file:
    dump(captions, file)

del file

result = [captions[key] for key in sorted(captions.keys())]
with open('captions.pickle', 'wb') as file:
    dump(result, file)

del file

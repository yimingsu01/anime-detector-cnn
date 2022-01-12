import cv2
import sys
import os.path
import argparse
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
# usage: python anime_detector.py -m fate_online.model -i test/test2.jpg -l lb.pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-m", "--model", required=True,
                help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True,
                help="path to label binarizer")
args = vars(ap.parse_args())

cascade_file = "lbpcascade_animeface.xml"

cascade = cv2.CascadeClassifier(cascade_file)
image = cv2.imread(args["image"], cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

faces = cascade.detectMultiScale(gray,
                                 # detector options
                                 scaleFactor=1.1,
                                 minNeighbors=5,
                                 minSize=(24, 24))

output = image.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cropped = output[y: y + h, x: x + w]

    cropped = cv2.resize(cropped, (100, 100))
    cropped = cropped.astype("float") / 255.0
    cropped = img_to_array(cropped)
    cropped = np.expand_dims(cropped, axis=0)

    proba = model.predict(cropped)[0]
    idx = np.argmax(proba)
    if proba[idx] > 0.01:
        label = lb.classes_[idx]
        label = "{}: {:.2f}%".format(label, proba[idx] * 100)
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)


cv2.imshow("AnimeFaceDetect", output)
cv2.waitKey(0)
cv2.imwrite("out.png", output)

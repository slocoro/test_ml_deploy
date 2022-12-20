# to push image to ECR:
# 0. authenticate docker to ECR
# aws ecr get-login-password --profile steven | docker login --username AWS --password-stdin 829131317437.dkr.ecr.eu-west-1.amazonaws.com/sleeves-lambda
# 1. build image from Dockerfile, -t aws-lambda is tag
# docker build --no-cache -t aws-lambda .
# 2. get image ID
# docker images
# 3. tag image
# docker tag 139058e8b4d5 829131317437.dkr.ecr.eu-west-1.amazonaws.com/sleeves-lambda:latest
# 4. push image to repository
# docker push 829131317437.dkr.ecr.eu-west-1.amazonaws.com/sleeves-lambda:latest


# follow this for guide on how to set up api gateway with lambda function
# https://www.youtube.com/watch?v=M91vXdjve7A

import json
import logging
import sys
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from pydantic import BaseModel

# create logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


def load_model():
    return tf.keras.models.load_model("model/sleeve_model")


def download_image(image_url):
    r = requests.get(image_url, stream=True)

    if r.status_code == 200:
        r.raw.decode_content = True

        return Image.open(BytesIO(r.content))


def load_and_decode_image(image_url):
    logger.info("Downloading image...")
    img = download_image(image_url)
    img = tf.convert_to_tensor(np.array(img))
    return tf.image.resize(img, [128, 128])


def add_batch_dimensions(image):
    return tf.expand_dims(image, axis=0)


normalize_image = tf.keras.layers.Rescaling(1.0 / 255)

model = load_model()


def handler(event, context):
    # TODO implement

    decode_event = json.loads(event["body"])
    url = decode_event["url"]

    image = load_and_decode_image(url)
    image = normalize_image(image)
    image = add_batch_dimensions(image)

    class_map = {
        0: "class_cap",
        1: "class_long",
        2: "class_raglan",
        3: "class_short",
        4: "class_sleeveless",
    }

    logger.info("Running inference...")
    prediction = model.predict(image, verbose=False)
    prediction_dict = {"prediction": class_map[prediction.argmax(axis=-1)[0]]}

    # AWS api gateway needs the response to be in this format for some reason
    response_object = {}
    response_object["statusCode"] = 200
    response_object["headers"] = {}
    response_object["body"] = json.dumps(prediction_dict)

    return response_object

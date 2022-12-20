import logging
import sys
from functools import lru_cache
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# create logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


app = FastAPI()


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


# this loads the model once for the first prediction then caches it
@lru_cache
def load_model():
    return tf.keras.models.load_model("model/sleeve_model")


@app.get("/")
async def get_root():
    return {"message": "Welcome to the sleeve prediction API."}


@app.post("/predict")
async def predict(image_url: ImageUrl):

    model = load_model()

    # Read image
    image = load_and_decode_image(image_url.url)
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
    return {"prediction": class_map[prediction.argmax(axis=-1)[0]]}


# command to run server manually:
# uvicorn src.predict:app --reload --port 80 --host 0.0.0.0

# to do:
# client code (done)
# logging (record inputs and outputs for each prediction)
#  need to create logging.conf file to make it work with FastAPI
# add unique prediction ID to predictions
# monitoring
# github actions to push image to ecr
# figure out how to make request using url only
# deploy to the cloud
# ec2 running image (done)
# ecs / fargate (done)
# two ec2 instances with load balancer in front of them
# api gateway + lambda (and route 53 for dns name)
# sagemaker
# tests for everything (try to use mock)
# have a database attached to the system??
# file with env variables

# deploy on ec2 manually:
# https://medium.com/bb-tutorials-and-thoughts/running-docker-containers-on-aws-ec2-9b17add53646
# (done)

# deployed on ecs fargate (done)

# deploy using lambda + api gateway (done)

# command to push image to docker hub
# 1. log into docker through cli
# docker login --username stevensallright
# 2. check docker images on computer
# docker images
# 3. take docker image ID
# 4. tag image
# docker tag 7ee0b36f3c32 stevensallright/test_ml_deploy-ml_deploy:latest
# docker push stevensallright/test_ml_deploy-ml_deploy
# "stevensallright" is docker hub username "test_ml_deploy-ml_deploy" is repo name "latest" is tag

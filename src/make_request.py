try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except Exception as e:
    print(e)

import logging
import os
import sys

from predict_client import ModelClient

# create logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # should this be moved to predict.py

    url = f'http://{os.getenv("MODEL_URL")}:{os.getenv("MODEL_PORT")}/'
    client = ModelClient(url)
    # client = ModelClient('http://ec2-34-245-59-46.eu-west-1.compute.amazonaws.com/')
    # client = ModelClient("http://34.242.248.8/")

    # image_url = 'https://cdna.lystit.com/520/650/n/photos/genteroma/3137a375/balmain--Black-Ribbed-Short-Dress-With-V-neck-And-Long-Sleeves.jpeg'
    # TODO: improve argument passing
    image_url = sys.argv[1]
    prediction = client.get_inference(image_url)

    logger.info(f"Predicted: {prediction}")

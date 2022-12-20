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

    # aws ec2 only
    # client = ModelClient('http://ec2-34-245-59-46.eu-west-1.compute.amazonaws.com/')
    # client = ModelClient("http://34.242.248.8/")

    # aws lambda + api gateway
    # client = ModelClient("https://59i476e4wf.execute-api.eu-west-1.amazonaws.com/testo/predict-length")

    # TODO: improve argument passing
    image_url = sys.argv[1]
    prediction = client.get_inference(image_url)

    logger.info(f"Predicted: {prediction}")

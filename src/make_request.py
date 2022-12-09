import logging
import sys

from predict_client import ModelClient

# create logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # should this be moved to predict.py
    client = ModelClient("http://localhost:80/")

    # image_url = 'https://cdna.lystit.com/520/650/n/photos/genteroma/3137a375/balmain--Black-Ribbed-Short-Dress-With-V-neck-And-Long-Sleeves.jpeg'
    # TODO: improve argument passing
    image_url = sys.argv[1]
    prediction = client.get_inference(image_url)

    logger.info(f"Predicted: {prediction}")

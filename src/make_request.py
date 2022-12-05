from predict_client import ModelClient
import logging
import sys

# create logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    client = ModelClient('http://localhost:80/')

    prediction = client.get_inference(
        'https://cdna.lystit.com/520/650/n/photos/johnlewis/afcc8f77/whistles-Ivory-Thelma-Wedding-Jumpsuit.jpeg')

    logger.info(f'Predicted: {prediction}')
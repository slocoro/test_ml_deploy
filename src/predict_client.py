import logging

import requests

logger = logging.getLogger(__name__)

# TODO: add error handling
class ModelClient(object):
    def __init__(self, url):
        self.url = url

    def get_inference(self, picture_url):

        logger.info("Getting inference...")
        r = requests.post(self.url + "predict/", json={"url": picture_url})

        return r.content

# from foo.bar import f2
from unittest.mock import patch
from predict import load_and_decode_image
from PIL import Image

@patch('predict.download_image')
def test_load_and_decode_image(download_image):

    download_image.return_value = Image.open('image/long_image_40.jpg')
    resized_image = load_and_decode_image('test_url')
    print(resized_image.shape)
    assert resized_image.shape == (128, 128, 3)

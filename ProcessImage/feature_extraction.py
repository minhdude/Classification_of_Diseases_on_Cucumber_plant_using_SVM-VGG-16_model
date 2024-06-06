from ProcessImage.load_image import load_and_preprocess_image
from ProcessImage.vgg_16 import vgg_model


def extract_features(img_path):
    img = load_and_preprocess_image(img_path)
    vgg_features = vgg_model.predict(img)
    vgg_features = vgg_features.flatten()
    return vgg_features

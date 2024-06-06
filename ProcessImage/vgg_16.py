from keras.applications.vgg16 import VGG16

vgg_model = VGG16(weights='imagenet', include_top=False)

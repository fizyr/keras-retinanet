from keras_retinanet.preprocessing.trassir_generator import *

generator = TrassirGenerator()

print(generator.size())
print(generator.num_classes())


print(generator.name_to_label('person'))
print(generator.label_to_name(0))

print(generator.image_aspect_ratio(10))

print(generator.load_image(10))
print(generator.load_annotations(10))

import tensorflow as tf
import tensorflow_datasets as tfds

dataset = tfds.load('oxford_iiit_pet', split='train', as_supervised=True)

def preprocess(image, label):
    image = tf.image.resize(image, (128, 128))
    image = image / 255.0
    return image, label

dataset = dataset.map(preprocess).batch(32).shuffle(1000)

for image_batch, label_batch in dataset.take(1):
    print("Image batch shape:", image_batch.shape)
    print("Label batch shape:", label_batch.shape)

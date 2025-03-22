import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet', split='train', with_info=True, as_supervised=True)

print(info)

for image, label in dataset.take(1):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

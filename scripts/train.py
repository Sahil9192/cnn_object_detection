from cnn_model import model
from preprocess_data import dataset

train_dataset = dataset.take(3000)
val_dataset = dataset.skip(3000)

model.fit(train_dataset, validation_data=val_dataset, epochs=5)
model.save("../models/pet_detector.h5")

# Keras-AudioDataGenerator
### Augmented Audio Data Generator for 1D-Convolutional Neural Networks 
######Note: For analogy with Image Data Generator, please read the [ImageDataGenerator](https://keras.io/preprocessing/image/) documentation.

The Audio Data Generator generates batches of audio data with real-time data augmentation.
Data is looped over in batches. This method enables audio augmentation is CPU while training in parallel on the GPU.

####Basic Usage:

```
from AudioDataGenerator import AudioDataGenerator

datagen = AudioDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                shift=.2,
                horizontal_flip=True,
                zca_whitening=True)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train,y_train,batch_size=32),
                    steps_per_epoch = len(x_train)/32,
                    epochs = 100)
```
# Keras-AudioDataGenerator
### Augmented Audio Data Generator for 1D-Convolutional Neural Networks 
###### Note: For analogy with Image Data Generator, please read the [ImageDataGenerator](https://keras.io/preprocessing/image/) documentation.

The Audio Data Generator generates batches of audio data with real-time data augmentation.
Data is looped over in batches. This method enables audio augmentation in CPU while training in parallel on the GPU.

## Basic Usage:

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

## Arguments:

The following are the currently available augmentation parameters.

**featurewise_center**: *Boolean*. Set input mean to 0 over the dataset, feature-wise.

**samplewise_center**: *Boolean*. Set each sample mean to 0.

**featurewise_std_normalization**: *Boolean*. Divide inputs by std of the dataset, feature-wise.

**samplewise_std_normalization**: *Boolean*. Divide each input by its std.

**zca_epsilon**: epsilon for ZCA whitening. Default is *1e-6*.

**zca_whitening**: *Boolean*. Apply ZCA whitening.
###### Note: Make sure to apply zca_whitening when using a separate validation set.
**roll_range**: *Float* (fraction of total sample length). Range for horizontal circular shifts.

**horizontal_flip**: *Boolean*. Randomly flip inputs horizontally.

**zoom_range**: *Float* (fraction of zoom) or range of [lower, upper] Uniform Distribution.

**noise**: `[mean,std,'Normal']` or `[lower,upper,'Uniform']`
           Add Random Additive noise. Noise is added to the data with a .5 probability.
           
**noiseSNR**: *Float* required SNR in dB. Noise is added to the data with a .5 probability(NotImplemented).

**shift**: *Float* (fraction of total sample length). Range for horizontal shifts.

**fill_mode**: One of `{"constant", "nearest", "reflect" or "wrap"}`.  Default is *'nearest'*. Points outside the boundaries of the input are filled according to the given mode:
- 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
- 'nearest':  aaaaaaaa|abcd|dddddddd
- 'reflect':  abcddcba|abcd|dcbaabcd
- 'wrap':  abcdabcd|abcd|abcdabcd

**cval**: *Float* or *Int*. Value used for points outside the boundaries when `fill_mode = "constant"`.

**rescale**: rescaling factor. Defaults to *None*. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (before applying any other transformation).

**preprocessing_function**: function that will be implied on each input.
- The function will run after the audio is resized and augmented.
- The function should take one argument:
  - one audio (Numpy tensor with rank 2),
  - should output a Numpy tensor with the same shape.

**brightness_range**: Tuple or list of *Float*. Range for adding random gain to the audio. Doesn't have any effect if normalization is performed.

**data_format**: One of *{"channels_first", "channels_last"}*. "channels_last" mode means that the audio should have shape `(samples, width, channels)`, "channels_first" mode means that the data should have shape `(samples, channels, width)`. It defaults to the `image_data_format` value found in your Keras config file at `~/.keras/keras.json`. If you never set it, then it will be "channels_last".

**validation_split**: *Float*. Fraction of data reserved for validation (strictly between 0 and 1).

## Methods:

### `.fit()` Method
Compute the internal data stats related to the data-dependent transformations, based on an array of sample data. Only required if featurewise_center or featurewise_std_normalization or zca_whitening is `True`.
```
datagen.fit(x_train,seed=1)
```
### `.flow()` Method
Creates an iterable with batches. The iterator iterates indefinitely, therefore loops need to be manually broken.
```
batch_size = 32
for epoch in range(epochs):
    batch = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True):
        model.fit(x_batch, y_batch)
        batch += 1
        if batch >= len(x_train) / 32:
            break
```

### `.flow_from_directory()` Method
Not Implemented

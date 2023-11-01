import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# Cargar el conjunto de datos de imágenes captcha
# Asegúrate de tener un conjunto de datos de imágenes captcha etiquetadas
# Este es solo un ejemplo, necesitarás ajustar esto a tu propio conjunto de datos
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'captcha_images/samples',
    target_size=(50, 50),
    batch_size=32,
    class_mode='categorical')

# Crear el modelo
model = Sequential()

# Añadir capas
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit_generator(train_generator, steps_per_epoch=15, epochs=5)
import tensorflow as tf
import cv2
import numpy as np

# Cargar la imagen y preprocesarla
image = cv2.imread('img/cameraman.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = cv2.resize(image_gray, (224, 224))
image_gray = np.expand_dims(image_gray, axis=-1) / 255.0

# Definir el modelo de detección de bordes
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', input_shape=(224, 224, 1))
])

# Cargar los pesos pre-entrenados (opcional)
# model.load_weights('edge_model.h5')

# Obtener la predicción de bordes
edges = model.predict(np.expand_dims(image_gray, axis=0))

# Obtener la imagen de bordes como un arreglo NumPy
edge_image = edges.squeeze()

# Aplicar umbral para resaltar los bordes
threshold = 0.5
edge_image[edge_image < threshold] = 0
edge_image[edge_image >= threshold] = 1

# Mostrar la imagen de bordes
cv2.imshow('Edges', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

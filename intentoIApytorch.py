import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

# Definir una red neuronal simple para detectar bordes
class EdgeDetector(nn.Module):
    def __init__(self):
        super(EdgeDetector, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

# Cargar la imagen y preprocesarla
image = cv2.imread('img/imagen.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0

# Crear una instancia del modelo de detección de bordes
model = EdgeDetector()

# Cargar los pesos pre-entrenados (opcional)
# model.load_state_dict(torch.load('edge_model.pth'))

# Obtener la predicción de bordes
with torch.no_grad():
    edges = model(image_tensor)

# Obtener la imagen de bordes como un arreglo NumPy
edge_image = edges.squeeze().numpy()

# Aplicar umbral para resaltar los bordes
threshold = 0.5
edge_image[edge_image < threshold] = 0
edge_image[edge_image >= threshold] = 1

# Mostrar la imagen de bordes
cv2.imshow('Edges', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

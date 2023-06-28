from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import base64

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obtener la imagen cargada desde el formulario
        image = request.files['image']
        image.save('uploaded_image.jpg')  # Guardar la imagen en el servidor

        # Aplicar detección de bordes con Canny utilizando OpenCV
        img = cv2.imread('uploaded_image.jpg', 0)
        edges = cv2.Canny(img, 100, 200)

        # Guardar los bordes resultantes en un objeto BytesIO
        _, buffer = cv2.imencode('.jpg', edges)
        modified_image_stream = buffer.tobytes()

        # Preprocesar la imagen para la detección de bordes con PyTorch
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Crear una instancia del modelo de detección de bordes
        model = EdgeDetector()

        # Obtener la predicción de bordes
        with torch.no_grad():
            edges_tensor = model(image_tensor)

        # Obtener la imagen de bordes como un arreglo NumPy
        edge_image = edges_tensor.squeeze().numpy()

        # Aplicar umbral para resaltar los bordes
        threshold = 0.5
        edge_image[edge_image < threshold] = 0
        edge_image[edge_image >= threshold] = 1

        # Guardar los bordes generados por la red neuronal en un objeto BytesIO
        _, additional_buffer = cv2.imencode('.jpg', edge_image * 255)
        additional_image_stream = additional_buffer.tobytes()

        return jsonify({
            'modified_image_url': f'data:image/jpeg;base64,{base64.b64encode(modified_image_stream).decode()}',
            'additional_image_url': f'data:image/jpeg;base64,{base64.b64encode(additional_image_stream).decode()}'
        })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

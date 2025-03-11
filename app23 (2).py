import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import csv
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Clasificador de Fauna Venezolana", page_icon="🐾", layout="wide")

# Agregar CSS personalizado
st.markdown(
    """
    <style>
        body {
            background-image: url('https://example.com/maiquetia_background.jpg'); /* URL de la imagen de Maiquetía */
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4 {
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }
        .animacion {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
        }
        .no-space {
            margin: 0;
            padding: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Cargar el modelo
model_path = "/content/drive/MyDrive/Modelos/modelo_vgg16_v4_16cat3.h5"
try:
    model = tf.keras.models.load_model(model_path)
except OSError as e:
    st.error(f"No se pudo cargar el modelo: {e}")

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Lista de nombres de clases
class_names = [
    'Perico Multicolor',
    'Tortuga Arrau',
    'Ballena de Aletas',
    'Gato Tigre',
    'Jaguar',
    'Jicotea',
    'Cotorra Cabeciamarilla',
    'Manatí de las Indias',
    'Mono Araña',
    'Mono Nocturno',
    'Oso Andino',
    'Oso Hormiguero Gigante',
    'Pato de Torrente',
    'Paují de Yelmo',
    'Tonina',
    'Venado Andino',
]

# Descripciones de cada animal
species_info = {
    "Perico Multicolor": "El perico multicolor se encuentra en regiones de América Central y América del Sur. Hay múltiples especies de pericos, cada una con características únicas.",
    "Tortuga Arrau": "Conocida por su caparazón distintivo, esta tortuga se encuentra principalmente en ríos amazónicos. En peligro de extinción.",
    "Ballena de Aletas": "Una de las especies más grandes de ballenas, puede crecer hasta 20 metros. Su estado de conservación es vulnerable.",
    "Gato Tigre": "Un pequeño felino que habita en zonas tropicales de América del Sur. Su población es muy reducida debido a la pérdida de hábitat.",
    "Jaguar": "El felino más poderoso de América, habita en bosques tropicales y manglares. Hay una única especie en peligro de extinción.",
    "Jicotea": "Conocida también como tortuga matamata, es una especie acuática que vive en ríos. Su población es estable, pero vulnerable.",
    "Cotorra Cabeciamarilla": "Esta cotorra colorida es endémica de las zonas boscosas de Venezuela. Su población se ha visto reducida por la deforestación.",
    "Manatí de las Indias": "También conocido como vaca marina, es un mamífero acuático que enfrenta amenazas por la pérdida de hábitat. Hay solo 2 especies reconocidas.",
    "Mono Araña": "Conocido por su inteligencia, este primate habita en los bosques tropicales de América del Sur. Hay varias especies en peligro.",
    "Mono Nocturno": "Este primate es activo durante la noche y tiene una población estable en algunas áreas. Es un indicador de salud ambiental.",
    "Oso Andino": "El único oso nativo de Sudamérica. Su población está disminuyendo debido a la caza y la pérdida de hábitat.",
    "Oso Hormiguero Gigante": "Este gran mamífero se alimenta de hormigas y termitas. Su población está debilitada por la caza y la deforestación.",
    "Pato de Torrente": "Una especie de pato que vive en ríos de montaña. Su población es estable, pero vulnerable a cambios en el hábitat.",
    "Paují de Yelmo": "Ave emblemática de Venezuela, se encuentra en zonas montañosas. Las poblaciones están disminuyendo debido a la caza.",
    "Tonina": "Este delfín de agua dulce vive en ríos de América del Sur. En corrupción debido a la pesca y la contaminación.",
    "Venado Andino": "Adaptado a la alta montaña, este venado es único de los Andes venezolanos. Su población es pequeña y vulnerable.",
}

confidence_threshold = 0.20

# Encabezado con colores venezolanos
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #002B7F, #FCD116, #E40303); padding: 20px; border-radius: 10px;">
        <h1 class="no-space" style="text-align: center;">Clasificador de Fauna Venezolana en Peligro</h1>
        <p style="text-align: center; font-size: 18px;">Identifica especies endémicas y en peligro en Venezuela</p>
    </div>
    """, unsafe_allow_html=True
)

# Sección animación (Lottie)
st.markdown('<div class="animacion">', unsafe_allow_html=True)
st.markdown('<lottie-player src="https://assets3.lottiefiles.com/packages/lf20_6hsvu07x.json" background="transparent" speed="1" style="width: 300px; height: 300px;" loop autoplay></lottie-player>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Sección informativa de especies
st.markdown(
    """
    <div style="background-color: rgba(0, 0, 0, 0.5); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <h3 style="text-align: center;">Especies que puede clasificar el modelo:</h3>
    </div>
    """, unsafe_allow_html=True
)

# Listar especies con descripciones
for idx, species in enumerate(class_names, start=1):
    st.markdown(f"<p style='text-align: center; font-size:16px;'> {idx}. {species}: {species_info[species]}</p>", unsafe_allow_html=True)

# Subir imagen desde la cámara
st.markdown("<h3 style='text-align: center;'>📸 Captura una imagen desde la cámara</h3>", unsafe_allow_html=True)
image_from_camera = st.camera_input("Captura una imagen")

# Subir imagen desde archivo
st.markdown("<h3 style='text-align: center;'>📁 O sube una imagen para analizar</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Combinar métodos de entrada de imagen
if image_from_camera is not None:
    image = Image.open(image_from_camera)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
else:
    image = None

# Lógica para visualizar y procesar la imagen
if image is not None:
    st.markdown("<h4 style='text-align: center;'>Imagen cargada:</h4>", unsafe_allow_html=True)
    st.image(image, caption="Vista previa", use_column_width=True)

    # Preprocesar imagen y predecir
    image_processed = preprocess_image(image)
    predictions = model.predict(image_processed)

    # Obtener probabilidad y clase predicha
    max_probability = np.max(predictions)
    predicted_index = np.argmax(predictions)

    if max_probability < confidence_threshold:
        st.markdown("<h2 style='color: red; text-align: center;'>⚠️ Animal no registrado</h2>", unsafe_allow_html=True)
    else:
        predicted_species = class_names[predicted_index]
        st.markdown(
            f"<h2 style='color: #002B7F; text-align: center;'>🦉 Categoría predicha: {predicted_species}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"<p style='text-align: center; font-size: 18px;'>", unsafe_allow_html=True)

        # Mostrar descripción de la especie predicha
        st.markdown(
            f"<div style='text-align: center;'> <h4>Descripción: {species_info[predicted_species]}</h4></div>",
            unsafe_allow_html=True
        )

        # Sección de feedback
        st.markdown(
            """
            <div style="background-color: transparent; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center;">
                <h4>¿Es correcta la clasificación?</h4>
            </div>
            """, unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Correcta"):
                log_feedback(predicted_species, f"{max_probability:.2f}", "correcta")
                st.success("¡Gracias por tu feedback!")
        with col2:
            if st.button("❌ Incorrecta"):
                log_feedback(predicted_species, f"{max_probability:.2f}", "incorrecta")
                st.error("¡Gracias por tu feedback!")

# Pie de página con créditos
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <hr>
    <p style="text-align: center; font-size: 14px;">
    Proyecto desarrollado con <b>TensorFlow</b> y <b>Streamlit</b> – Especial para la fauna venezolana.
    </p>
    """, unsafe_allow_html=True
)

# Cargar el script para Lottie
st.markdown(
    """
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.6/lottie.min.js"></script>
    """, unsafe_allow_html=True
)

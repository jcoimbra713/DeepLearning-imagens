import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf

# Carregar o modelo salvo
model = tf.keras.models.load_model('cifar10_model.h5')

# Classes do CIFAR-10 em português
class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Função principal da aplicação Streamlit
def main():
    st.title("IA-CRIADA PARA PREVER IMAGENS - POR JOÃO COIMBRA")
    st.write("Faça o upload de uma imagem para classificá-la.")

    # Upload da imagem
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["JPG", "GIF", "PNG", "SVG", "PSD", "WEBP", "RAW", "TIFF", "BMP", "JFIF"])

    if uploaded_file is not None:
        # Salvar a imagem carregada
        img_path = "uploaded_image." + uploaded_file.name.split('.')[-1]
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Pré-processar a imagem
        img_array = load_and_preprocess_image(img_path)

        # Fazer a predição
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Mostrar a classe predita
        st.write(f'Classe predita: {class_names[predicted_class]}')

        # Carregar e mostrar a imagem original
        original_image = image.load_img(img_path)
        st.image(original_image, caption=f'Classe predita: {class_names[predicted_class]}', use_column_width=True)

if __name__ == '__main__':
    main()







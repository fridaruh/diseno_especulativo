import streamlit as st
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset
from huggingface_hub import InferenceClient
import time

import streamlit as st

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="👩‍👩‍👦",
    
)

#Obtenemos la variable de entorno para habilitar el transfer de Hugging Face
hf_transfer_enabled = os.getenv('HF_HUB_ENABLE_HF_TRANSFER')

#Instanciamos el cliente de Hugging Face

client = InferenceClient(api_key="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")


#Llamada a la API de Hugging Face
def obtener_respuesta(prompt):
    output = ""
    for message in client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You're a simulated person that responds in Spanish in 100 words, straight to the point"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        stream=True
    ):
        if message.choices[0].delta.content is not None:
            output += message.choices[0].delta.content
    return output

#Procesamiento de las descripciones de las personas
import streamlit as st
import pandas as pd

def procesar_descripciones(df, prompt_template):
    resultados = []
    total = len(df)

    progress_text = "Procesando personas. Por favor, espere."
    my_bar = st.progress(0, text=progress_text)

    for index, row in enumerate(df.iterrows()):
        persona = row[1]['persona']
        prompt = prompt_template.format(persona=persona)

        try:
            output = obtener_respuesta(prompt)
            resultados.append({'persona': persona, 'output': output})
        except Exception as e:
            st.error(f"Error procesando persona: {persona}. Error: {str(e)}")
            resultados.append({'persona': persona, 'output': 'Error en el procesamiento'})

        # Actualizar la barra de progreso
        progress = min(100 * (index + 1) / total, 100)
        my_bar.progress(int(progress), text=f"Procesando: {progress:.1f}%")

    my_bar.empty()
    return pd.DataFrame(resultados)


@st.cache_data
def cargar_dataset():
    """
    Carga el dataset desde la URL y lo almacena en caché.
    """
    return pd.read_csv("https://raw.githubusercontent.com/fridaruh/diseno_especulativo/refs/heads/master/fine_personas_100k.csv?token=GHSAT0AAAAAACYRTNSVIGEZYKHBYZPCQWAKZYULDUQ")

def obtener_muestra(df, n=15):
    """
    Obtiene una muestra aleatoria del dataframe.
    """
    sample = df['persona'].sample(n)
    return pd.DataFrame(sample, columns=['persona'])

# Uso de las funciones
df = cargar_dataset()
sample = obtener_muestra(df)


def main():
    st.title("Diseño Especulativo")

    st.write("Selecciona lo que te gustaría observar:")
    
    options = ["Reacción", "Interés", "Preocupaciones", "Nivel de aceptación", "Jiribilla"]
    situacion_a_observar = st.selectbox("", options)

    st.write("¿Cuál es el proyecto o iniciativa que te gustaría testear?")
    project = st.text_area("")

    if st.button("Enviar"):

        prompt_template = """
Simulate how the following consumer segment would react to the launch of a new wearable technology product
(e.g., a smartwatch) that monitors health, focusing on disease prevention and wellness:
{persona}
"""

        # Asumiendo que ya tienes tu DataFrame 'df' con la columna 'persona'
        df_resultados = procesar_descripciones(sample, prompt_template)
        st.table(df_resultados.head())


    





if __name__ == "__main__":
    main()    
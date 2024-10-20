import streamlit as st
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset
from huggingface_hub import InferenceClient
import time

import streamlit as st

st.set_page_config(
    page_title="Simulador de impacto cultural",
    page_icon="👩‍👩‍👦",
    
)

#Instanciamos el cliente de Hugging Face

client = InferenceClient(api_key="")


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
    return pd.read_csv("https://raw.githubusercontent.com/fridaruh/diseno_especulativo/refs/heads/master/fine_personas_100k.csv")

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
    st.title("Simulador de impacto cultural")
    st.markdown("""
    *"El comportamiento humano es en esencia cultural, establece las pautas de lo que es apropiado, aceptado o inapropiado.
                Así como nuestras formas de interactuar y tomar decisiones." 
                                    - Dagoberto Páramo*
    """)

    st.write("Selecciona lo que te gustaría observar:")
    
    options = ["Reacción", "Interés", "Preocupaciones", "Nivel de aceptación", "Jiribilla"]
    situacion_a_observar = st.selectbox("", options)

    st.write("¿Cuál es el proyecto o iniciativa que te gustaría testear?")

    project = st.text_area("¿Cuál es el proyecto o iniciativa que te gustaría testear?",
                           "Ejemplo: 'Proyecto de desarrollo de transporte eléctrico elevado (CableBus) en la zona urbana de la Ciudad, en conexión con el norte de la Ciudad, colonia San Sebastian Sedas con el Centro histórico de la Ciudad de Oaxaca'")

    st.write("¿En qué Estado te gustaría probarlo?")

     #Define the options and their descriptions
    cultural_options = {
        "CDMX": "Bajo sentido de pertenencia y participación en las instituciones políticas y económicas, se sienten desconectadas del sistema de gobierno, afectando la cohesión y participación ciudadana.",
        "Veracruz": "Se caracteriza por su hospitalidad, orgullo en sus tradiciones, resiliencia ante las dificultades y un enfoque fuerte en la familia y la comunidad, celebrando su diversidad cultural.",
        "Oaxaca": "La comunidad es central en la vida social. El trabajo comunitario refleja un fuerte sentido de solidaridad y colaboración. Son conocidos por su amabilidad y hospitalidad, creando un ambiente acogedor. La relación con la tierra y las tradiciones agrícolas es fundamental. Los festivales celebran las cosechas y el ciclo agrícola, reforzando los vínculos con la naturaleza y la cultura.",
        "Monterrey": "Se distinguen por valores como la perseverancia, la familia, el trabajo y la solidaridad. Existe una fuerte ética del trabajo y un enfoque en el éxito personal y profesional. Suelen valorar el esfuerzo y la dedicación como medios para alcanzar objetivos. La proximidad a Estados Unidos ha influido en las creencias y costumbres, introduciendo elementos de cultura norteamericana que se mezclan con las tradiciones locales.",
        "Yucatán": "Sienten profundamente orgullosos de su identidad, con un fuerte apego a su tierra, su historia y sus héroes locales. Valoran la hospitalidad, la solidaridad comunitaria y el respeto a las tradiciones. Prefieren la sencillez y la belleza natural de su entorno, como los crepúsculos y la brisa del mar, que forman parte de su identidad colectiva."
    }

    # Create the select box
    selected_state = st.selectbox(
    "Selecciona una región para el ajuste cultural:",
    options=list(cultural_options.keys())
)

    # Assign the description to cultural_tuning based on the selected state
    cultural_tuning = cultural_options[selected_state]

    if st.button("Enviar"):


        prompt_inicial = """
        Simulate how the following specific person {persona}"""
        
        prompt_complementario = """ 
        
        who lives in a cultural context where: {cultural_tuning}. How this person would react to the launch of {project}?

        Focus specifically on the {situacion_a_observar} of this specific person regarding the project.
        Provide a detailed analysis of their likely {situacion_a_observar}, considering the unique characteristics and background of this consumer profile.

        Guidelines for the analysis based on the selected focus:
        - Reacción: Describe the immediate emotional and cognitive response of the consumer.
        - Interés: Evaluate the level and nature of interest the consumer might have in the project.
        - Preocupaciones: Identify and explain potential concerns or reservations the consumer might have.
        - Nivel de aceptación: Assess how likely the consumer is to accept and adopt the project.
        - Jiribilla: Explore any unique or unexpected ways this consumer might interact with or perceive the project.

        Ensure your response is tailored to the specific consumer profile and the nature of the project described."""
        prompt_complementario = prompt_complementario.format(
        cultural_tuning=cultural_tuning,
        project=project,
        situacion_a_observar=situacion_a_observar
         )
        
        # Asumiendo que ya tienes tu DataFrame 'df' con la columna 'persona'
        df_resultados = procesar_descripciones(sample, prompt_inicial + prompt_complementario)

        # Inicialización del DataFrame en session_state
        if 'df_resultados' not in st.session_state:
            st.session_state['df_resultados'] = None

        st.table(df_resultados.head())
        st.download_button(
            label="Download data as CSV",
            data=df_resultados.to_csv(index=False).encode('utf-8'),
            file_name="large_df.csv",
            mime="text/csv",
        )
        
        st.session_state['df_resultados'] = df_resultados

if __name__ == "__main__":
    main()    

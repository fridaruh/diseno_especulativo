import streamlit as st
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset
from huggingface_hub import InferenceClient
import time

st.set_page_config(
    page_title="Simulador de impacto cultural",
    page_icon="👩‍👩‍👦",
)

@st.cache_resource
def get_hf_client():
    try:
        return InferenceClient(api_key=st.secrets["huggingface_api_key"])
    except Exception as e:
        st.error(f"Error al inicializar el cliente de Hugging Face: {e}")
        return None

client = get_hf_client()

# Función para obtener la respuesta de la API de Hugging Face
def obtener_respuesta(prompt):
    output = ""
    try:
        for message in client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "Eres una persona simulada que responde en español en 100 palabras, directo al punto."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            stream=True
        ):
            if message.choices[0].delta.content is not None:
                output += message.choices[0].delta.content
    except Exception as e:
        st.error(f"Error al obtener respuesta de la API: {str(e)}")
        output = 'Error al obtener respuesta de la API'
    return output

# Procesamiento de las descripciones de las personas
def procesar_descripciones(df, prompt_template):
    resultados = []
    total = len(df)

    if total == 0:
        st.warning("El DataFrame está vacío. No hay personas para procesar.")
        return pd.DataFrame()

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
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/fridaruh/diseno_especulativo/refs/heads/master/fine_personas_100k.csv")
        return df
    except Exception as e:
        st.error(f"Error al cargar el dataset: {str(e)}")
        return pd.DataFrame()  # Devuelve un DataFrame vacío en caso de error

def obtener_muestra(df, n=15):
    """
    Obtiene una muestra aleatoria del dataframe.
    """
    try:
        if df.empty:
            st.warning("El DataFrame está vacío. No se puede obtener una muestra.")
            return pd.DataFrame(columns=['persona'])
        sample = df['persona'].sample(n)
        return pd.DataFrame(sample, columns=['persona'])
    except Exception as e:
        st.error(f"Error al obtener la muestra: {str(e)}")
        return pd.DataFrame(columns=['persona'])

# Uso de las funciones
df = cargar_dataset()
sample = obtener_muestra(df)

def main():
    st.title("Simulador de impacto de políticas públicas")
    st.markdown("""
    *"El comportamiento humano es en esencia cultural, establece las pautas de lo que es apropiado, aceptado o inapropiado.
                Así como nuestras formas de interactuar y tomar decisiones." 
                                    - Dagoberto Páramo*
    """)

    st.write("Selecciona lo que te gustaría observar:")
    
    options = ["Preocupaciones","Reacción", "Interés", "Nivel de aceptación", "Jiribilla"]
    situacion_a_observar = st.selectbox("", options)

    st.write("¿Cuál es el proyecto o iniciativa que te gustaría testear?")
    
    # Validación de entrada del usuario
    project = st.text_area(
        "Describe el proyecto o iniciativa de manera clara y concisa",
        "Ejemplo: 'Proyecto de desarrollo de transporte eléctrico elevado (CableBus) en la zona urbana de la Ciudad, en conexión con el norte de la Ciudad, colonia San Sebastian Sedas con el Centro histórico de la Ciudad de Oaxaca'"
    )
    if not project.strip():
        st.error("Por favor, ingresa un proyecto o iniciativa para testear.")
        return  # Detiene la ejecución si el proyecto está vacío

    st.write("¿En qué Estado te gustaría probarlo?")
    
    # Definición de las opciones y sus descripciones
    cultural_options = {
        
        "CDMX": "Bajo sentido de pertenencia y participación en las instituciones políticas y económicas, se sienten desconectadas del sistema de gobierno, afectando la cohesión y participación ciudadana.",
        "Veracruz": "Se caracteriza por su hospitalidad, orgullo en sus tradiciones, resiliencia ante las dificultades y un enfoque fuerte en la familia y la comunidad, celebrando su diversidad cultural.",
        "Monterrey": "Se distinguen por valores como la perseverancia, la familia, el trabajo y la solidaridad. Existe una fuerte ética del trabajo y un enfoque en el éxito personal y profesional. Suelen valorar el esfuerzo y la dedicación como medios para alcanzar objetivos. La proximidad a Estados Unidos ha influido en las creencias y costumbres, introduciendo elementos de cultura norteamericana que se mezclan con las tradiciones locales.",
        "Yucatán": "Sienten profundamente orgullosos de su identidad, con un fuerte apego a su tierra, su historia y sus héroes locales. Valoran la hospitalidad, la solidaridad comunitaria y el respeto a las tradiciones. Prefieren la sencillez y la belleza natural de su entorno, como los crepúsculos y la brisa del mar, que forman parte de su identidad colectiva."
    }

    # Creación del select box
    selected_state = st.selectbox(
        "Selecciona una región para el ajuste cultural:",
        options=list(cultural_options.keys())
    )

    # Asignación de la descripción según el estado seleccionado
    cultural_tuning = cultural_options[selected_state]

    if st.button("Enviar"):
        try:
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

            # Verificación de que la muestra no esté vacía
            if sample.empty:
                st.warning("No se pudo obtener una muestra de datos.")
                return

            # Procesamiento de las descripciones
            df_resultados = procesar_descripciones(sample, prompt_inicial + prompt_complementario)

            if df_resultados.empty:
                st.warning("No se obtuvieron resultados del procesamiento.")
                return

            # Inicialización del DataFrame en session_state
            if 'df_resultados' not in st.session_state:
                st.session_state['df_resultados'] = None

            st.table(df_resultados.head())
            st.download_button(
                label="Descargar datos como CSV",
                data=df_resultados.to_csv(index=False).encode('utf-8'),
                file_name="resultados.csv",
                mime="text/csv",
            )

            st.session_state['df_resultados'] = df_resultados

        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")

if __name__ == "__main__":
    main()

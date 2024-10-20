import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import InferenceClient
import plotly.express as px
import plotly.graph_objects as go

# Descarga de datos de NLTK
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"Error al descargar los datos de NLTK: {e}")

# Configuración del cliente de Hugging Face
@st.cache_resource
def get_hf_client():
    try:
        return InferenceClient(api_key=st.secrets["huggingface_api_key"])
    except Exception as e:
        st.error(f"Error al inicializar el cliente de Hugging Face: {e}")
        return None

client = get_hf_client()

# Función para obtener el sentimiento
def obtener_sentimiento(prompt):
    output = ""
    try:
        for message in client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": """
                
                You are a sentiment analysis tool. Your task is to classify the sentiment of the given text into exactly one of these three categories: Positive, Negative, or Neutral. 

                Rules:
                1. Respond ONLY with "Positive", "Negative", or "Neutral".
                2. Do not provide any explanation or additional text.
                3. If the sentiment is mixed or unclear, choose "Neutral".
                4. Consider the overall tone of the entire text, not just individual words.
                5. "Positive" indicates clearly good feelings or opinions.
                6. "Negative" indicates clearly bad feelings or opinions.
                7. "Neutral" is for factual statements, mixed sentiments, or unclear cases.

                 """},
                {"role": "user", "content": f'Analyze the sentiment of the following text and respond with only one word: {prompt}'}
            ],
            max_tokens=500,
            stream=True
        ):
            if message.choices[0].delta.content is not None:
                output += message.choices[0].delta.content
    except Exception as e:
        st.error(f"Error al obtener el sentimiento: {e}")
        output = "Neutral"  # Valor por defecto en caso de error
    return output.strip()

def get_sentiment_percentage(df, sentiment):
    if sentiment in df['sentiment'].values:
        return df.loc[df['sentiment'] == sentiment, 'percentage'].values[0]
    return 0.0

# Función para procesar texto
def procesar_texto(df):
    try:
        data = df['output'].to_list()
        texto_unido = ' '.join(texto.strip('"') for texto in data)
        return texto_unido
    except Exception as e:
        st.error(f"Error al procesar el texto: {e}")
        return ""

# Aplicación Streamlit
st.title("Análisis de sentimiento de la iniciativa")
st.write("El análisis puede tomar hasta 3 minutos en terminar de realizarse")

# Verificar si 'df_resultados' existe en el estado de la sesión
if 'df_resultados' not in st.session_state or st.session_state['df_resultados'] is None:
    st.error("No se encontró 'df_resultados' en el estado de la sesión. Por favor, ejecuta primero el proceso anterior.")
else:
    # Recuperar el DataFrame de la sesión
    df_resultados = st.session_state['df_resultados']

    # Mostrar una vista previa del DataFrame
    st.write("Vista previa de los datos:")
    st.dataframe(df_resultados.head())

    if 'output' not in df_resultados.columns:
        st.error("El DataFrame no contiene la columna 'output'.")
    else:
        output = df_resultados['output']
        output = pd.DataFrame(output, columns=['output'])

        if client is None:
            st.error("No se puede proceder sin el cliente de Hugging Face.")
        else:
            # Usando st.status en lugar de st.spinner
            status_container = st.status("Analizando sentimientos...", state="running")

            try:
                tqdm.pandas(desc="Analizando sentimientos")
                output['sentiment'] = output['output'].progress_apply(obtener_sentimiento)
                status_container.update(label="¡Análisis de sentimiento completado!", state="complete")
            except Exception as e:
                status_container.update(label=f"Error durante el análisis de sentimiento: {e}", state="error")
                st.stop()

            # Continuar con el resto del código si no hubo errores
            # Calcular la distribución de sentimientos
            sentiment_counts = output['sentiment'].value_counts()
            sentiment_percentages = sentiment_counts / len(output) * 100

            # Crear un DataFrame para la visualización
            sentiment_df = pd.DataFrame({
                'sentiment': sentiment_counts.index,
                'count': sentiment_counts.values,
                'percentage': sentiment_percentages.values
            })

            st.title("Análisis de Sentimientos")

            # 1. Métricas Destacadas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentimiento Positivo", f"{get_sentiment_percentage(sentiment_df, 'Positive'):.1f}%")
            with col2:
                st.metric("Sentimiento Neutral", f"{get_sentiment_percentage(sentiment_df, 'Neutral'):.1f}%")
            with col3:
                st.metric("Sentimiento Negativo", f"{get_sentiment_percentage(sentiment_df, 'Negative'):.1f}%")

            # Gráfico de Pastel
            st.subheader("Proporción de Sentimientos")
            try:
                fig_pie = px.pie(sentiment_df, values='count', names='sentiment', 
                                 color='sentiment',
                                 color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFA500', 'Negative': '#F44336'},
                                 hole=0.3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Error al crear el gráfico de pastel: {e}")

            # Filtrar tablas por sentimiento
            positive_table = output[output['sentiment'] == 'Positive']
            negative_table = output[output['sentiment'] == 'Negative']
            neutral_table = output[output['sentiment'] == 'Neutral']

            # Generar nubes de palabras
            stop_words_n = set(stopwords.words('spanish'))

            st.subheader("Nube de Palabras Positivas")
            positive_text = procesar_texto(positive_table)
            if positive_text:
                try:
                    wc_positive = WordCloud(background_color='white', colormap='Greens',
                                            stopwords=stop_words_n, width=800, height=500).generate(positive_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_positive, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al generar la nube de palabras positivas: {e}")
            else:
                st.write("No hay texto positivo para generar la nube de palabras.")

            st.subheader("Nube de Palabras Negativas")
            negative_text = procesar_texto(negative_table)
            if negative_text:
                try:
                    wc_negative = WordCloud(background_color='white', colormap='Reds',
                                            stopwords=stop_words_n, width=800, height=500).generate(negative_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_negative, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al generar la nube de palabras negativas: {e}")
            else:
                st.write("No hay texto negativo para generar la nube de palabras.")

            st.subheader("Nube de Palabras Neutrales")
            neutral_text = procesar_texto(neutral_table)
            if neutral_text:
                try:
                    wc_neutral = WordCloud(background_color='white', colormap='Blues',
                                           stopwords=stop_words_n, width=800, height=500).generate(neutral_text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_neutral, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al generar la nube de palabras neutrales: {e}")
            else:
                st.write("No hay texto neutral para generar la nube de palabras.")

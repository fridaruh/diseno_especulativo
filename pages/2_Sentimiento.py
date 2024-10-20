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

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Hugging Face client setup
@st.cache_resource
def get_hf_client():
    return InferenceClient(api_key="")

client = get_hf_client()

# Function to get sentiment
def obtener_sentimiento(prompt):
    output = ""
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
    return output.strip()

def get_sentiment_percentage(df, sentiment):
    if sentiment in df['sentiment'].values:
        return df.loc[df['sentiment'] == sentiment, 'percentage'].values[0]
    return 0.0

# Function to process text
def procesar_texto(df):
    data = df['output'].to_list()
    texto_unido = ' '.join(texto.strip('"') for texto in data)
    return texto_unido

# Streamlit app
st.title("Análisis de sentimiento de la iniciativa")

#Recuperar el dataframe de la sesión
df_resultados = st.session_state['df_resultados']

# Display the DataFrame
st.write("Data Preview:")


output = st.session_state.df_resultados['output']
output = pd.DataFrame(output, columns=['output'])

with st.spinner("Analyzing sentiments..."):
    tqdm.pandas(desc="Analyzing sentiments")
    output['sentiment'] = output['output'].progress_apply(obtener_sentimiento)

st.write("Sentiment Analysis Complete!")

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
fig_pie = px.pie(sentiment_df, values='count', names='sentiment', 
                 color='sentiment',
                 color_discrete_map={'Positive': '#4CAF50', 'Neutral': '#FFA500', 'Negative': '#F44336'},
                 hole=0.3)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# Filter tables by sentiment
positive_table = output[output['sentiment'] == 'Positive']
negative_table = output[output['sentiment'] == 'Negative']
neutral_table = output[output['sentiment'] == 'Neutral']

# Process text for each sentiment
positive_text = procesar_texto(positive_table)
negative_text = procesar_texto(negative_table)
neutral_text = procesar_texto(neutral_table)

# Generate word clouds
stop_words_n = set(stopwords.words('spanish'))

st.subheader("Positive Word Cloud")
wc_positive = WordCloud(background_color='white', colormap='binary',
                        stopwords=stop_words_n, width=800, height=500).generate(positive_text)
fig, ax = plt.subplots()
ax.imshow(wc_positive)
ax.axis("off")
st.pyplot(fig)

st.subheader("Negative Word Cloud")
wc_negative = WordCloud(background_color='white', colormap='binary',
                        stopwords=stop_words_n, width=800, height=500).generate(negative_text)
fig, ax = plt.subplots()
ax.imshow(wc_negative)
ax.axis("off")
st.pyplot(fig)

st.subheader("Neutral Word Cloud")
wc_neutral = WordCloud(background_color='white', colormap='binary',
                       stopwords=stop_words_n, width=800, height=500).generate(neutral_text)
fig, ax = plt.subplots()
ax.imshow(wc_neutral)
ax.axis("off")
st.pyplot(fig)

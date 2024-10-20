# Simulador de Impacto de políticas públicas

Este proyecto es una aplicación web desarrollada con Streamlit que simula cómo diferentes perfiles de personas podrían reaccionar ante una nueva iniciativa o proyecto de políticas públicas, considerando factores culturales y emocionales.

## Descripción

La aplicación permite a los usuarios:

- Seleccionar una región específica para ajustar el contexto cultural.
- Especificar un proyecto o iniciativa para testear.
- Elegir un aspecto particular para observar (e.g., Reacción, Interés, Preocupaciones).
- Generar y analizar respuestas simuladas de personas basadas en descripciones detalladas.
- Realizar análisis de sentimiento sobre las respuestas generadas.
- Visualizar los resultados a través de métricas destacadas, gráficos y nubes de palabras.

## Características

- **Interfaz Interactiva**: Utiliza Streamlit para proporcionar una experiencia de usuario amigable.
- **Integración con Hugging Face**: Se conecta con modelos de lenguaje para generar respuestas y análisis de sentimiento utilizando el dataset FinePersona: https://huggingface.co/datasets/argilla/FinePersonas-v0.1
- **Análisis de Sentimiento**: Clasifica las respuestas en positivas, negativas o neutrales.
- **Visualizaciones**: Incluye gráficos y nubes de palabras para representar los resultados.

## Requisitos Previos

- Python 3.7 o superior.
- Clave API de Hugging Face.

## Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/fridaruh/diseno_especulativo.git
cd diseno_especulativo

2. **Crea un entorno virtual (opcional pero recomendado):**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Descarga los recursos de NLTK:**

   En tu terminal o en un script Python, ejecuta:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

5. **Configuración**

   - Clave API de Hugging Face:

- Crea un archivo llamado secrets.toml en el directorio .streamlit de tu proyecto (crea el directorio si no existe):

.streamlit/
  secrets.toml

- Agrega tu clave API al archivo secrets.toml:

[general]
huggingface_api_key = "TU_CLAVE_API"

##Uso
Ejecuta la aplicación de Streamlit:

```bash
streamlit run Hello.py
```
## Estructura del Proyecto

- **app.py**: Archivo principal que contiene la lógica de la aplicación.
- **analysis.py**: Contiene el código para el análisis de sentimientos y visualizaciones.
- **requirements.txt**: Lista de dependencias necesarias para ejecutar la aplicación.
- **.streamlit/secrets.toml**: Archivo que contiene las claves y secretos necesarios (no incluido en el repositorio).

## Contribuyendo
Si deseas contribuir, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una rama con tu nueva característica (git checkout -b feature/nueva_caracteristica).
3. Haz commit de tus cambios (git commit -am 'Agrega nueva característica').
4. Haz push a la rama (git push origin feature/nueva_caracteristica).
5. Abre un Pull Request.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

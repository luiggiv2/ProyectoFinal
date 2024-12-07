# Implementación y aprovechamiento de un Modelo de Lenguaje Grande (LLM) para análisis de correos electrónicos y detección de phishing

El presente informe aborda el desarrollo e implementación de un sistema de detección de correos electrónicos de phishing utilizando modelos de lenguaje grande (Large Language Models, LLMs) y Python.

Datos recientes muestran alrededor de 1,9 millones de ataques de phishing en el último año, en el tercer semestre del 2024 se han identificado 932.923 ataques de phishing frente a los 877.536 observados en el segundo trimestre del mismo año, el panorama de las estafas está más activo que nunca, exigiendo defensas innovadoras.

El FBI de Estados Unidos recibió casi, 300000 denuncias en 2023, lo que convierte al phishing en el delito cibernético más denunciado en los últimos 5 años. El phishing es una amenaza global.

**Fuente:**
https://apwg.org/trendsreports/ <br>
https://docs.apwg.org/reports/apwg_trends_report_q2_2024.pdf <br>
https://static1.squarespace.com/static/63dbf2b9075aa2535887e365/t/66cde404c8345e766972319c/1724769286084/PhishingLandscape2024.pdf<br>

## Preparación de entorno
Se utilizará Google Colab, ya que es una plataforma gratuita en la nube que ofrece acceso a los recursos de la GPU; óptimo para realizar análisis basados en modelos LLM. <br>
https://colab.research.google.com

## Arquitectura Implementada

<img width="730" alt="Captura de pantalla 2024-12-06 a la(s) 21 16 22" src="https://github.com/user-attachments/assets/a5d1ff7a-bed4-421a-a281-70003e9739ac">

## Configuración

Primero, se realiza la instalación de Ollama y configurar nuestro entorno. La configuración es exactamente la misma que hicimos en la colaboración anterior. La siguiente línea de código es un atajo para descargar e ejecutar el script de instalación de Ollama en una sola línea de comando.

```
# Se realiza la instalación de Ollama
!curl https://ollama.ai/install.sh | sh
```

A continuación, se configura las variables de entorno necesarias e iniciaremos el servidor Ollama.

```python
# Se configura las variables de entorno necesarias e iniciaremos el servidor Ollama
import os
import asyncio
import threading
import time

# NB: Es posible que necesites configurarlos y hacer que cuda funcione según el backend que estés ejecutando.
# Se establezce la variable de entorno para la biblioteca NVIDIA
# Se establezce las variables de entorno para CUDA
os.environ['PATH'] += ':/usr/local/cuda/bin'
# Se establezca LD_LIBRARY_PATH para incluir los directorios /usr/lib64-nvidia y CUDA lib
os.environ['LD_LIBRARY_PATH'] = '/usr/lib64-nvidia:/usr/local/cuda/lib64'

# Se establece funciones auxiliares para ejecutar el servidor Ollama
async def run_process(cmd, stdout=None, stderr=None):
    print('>>> starting', *cmd)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=stdout or asyncio.subprocess.PIPE,
        stderr=stderr or asyncio.subprocess.PIPE
    )

    if stdout is None and stderr is None:
        async def pipe(lines):
            async for line in lines:
                print(line.decode().strip())

        await asyncio.gather(
            pipe(process.stdout),
            pipe(process.stderr),
        )
    else:
        await process.wait()

async def start_ollama_serve():
    await run_process(['ollama', 'serve'],
                      stdout=open(os.devnull, 'w'),
                      stderr=open(os.devnull, 'w'))

# Se inicia el servidor Ollama
def run_async_in_thread(loop, coro):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

# Se crea un nuevo bucle de eventos que se ejecutará en un nuevo hilo
new_loop = asyncio.new_event_loop()

# Se inicia el servicio de Ollama en un hilo separado para que la celda no bloquee la ejecución
thread = threading.Thread(target=run_async_in_thread, args=(new_loop, start_ollama_serve()))
thread.start()

# Se espera 5 segundos para que olama se cargue
time.sleep(5)
```

Luego se instalarán dependencias necesarias, en este caso paquetes Python que serán necesarios.

```
# Luego instalaremos los paquetes de Python necesarios
%pip install openai pydantic instructor ipywidgets beautifulsoup4 pandas tqdm chardet matplotlib seaborn wordcloud
```

Se continúa descargando el modelo LLM, en este se elige Gemma2 desarrollado por Google AI ya que se caracteriza por su alto rendimiento y eficiencia. Tiene 9 mil millones de parámetros y ha sido entrenado para seguir intrucciones y realizar tareas de manera efectiva.

```
# Descargamos el modelo Google gemma2 en su versión 9B,
# conocido por su fuerte rendimiento (especialmente en tareas de extracción de
# datos) con respecto a su tamaño.
import os
# Establecer la variable MODELO
OLLAMA_MODEL = "gemma2:9b-instruct-q4_K_M"

# Ejecuta el comando de extracción de ollama
! ollama pull {OLLAMA_MODEL}
```

Se define los modelos de datos para estructurar los resultados del análisis de phishing.

```python
# Se define los modelos de Pydantic para estructurar los resultados del
# análisis de phishing
from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class PhishingProbability(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class SuspiciousElement(BaseModel):
    element: str
    reason: str

class SimplePhishingAnalysis(BaseModel):
    is_potential_phishing: bool
    phishing_probability: PhishingProbability
    suspicious_elements: List[SuspiciousElement]
    recommended_actions: List[str]
    explanation: str
```

Se implementan funciones de análisis de correo, se crea una función para analizar correos electrónicos utilizando nuestro LLM.

```python
# Se crea la función de análisis para analizar correos electrónicos
# utilizando LLM

from openai import OpenAI
import httpx
import instructor

# Define OLLAMA_BASE_URL globalmente (o como una variable de entorno)
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"  # Actualice con su dirección de servidor Ollama actual

def analyze_email(email_content: str) -> SimplePhishingAnalysis:
    http_client = httpx.Client()
    client = instructor.from_openai(
      OpenAI(
          base_url=OLLAMA_BASE_URL,
          api_key="ollama",  # Requerida, pero no utilizada
          http_client=http_client  # Especifica el cliente HTTP
      ),
      mode=instructor.Mode.JSON,
  )

    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an advanced AI assistant specialized in cybersecurity, particularly in detecting and analyzing phishing attempts in emails. Your task is to analyze the provided email content and metadata to determine if it's a potential phishing attempt. You must provide your analysis in a structured format that matches the model.",
            },
            {
                "role": "user",
                "content": email_content,
            },
        ],
        response_model=SimplePhishingAnalysis,
    )
    return resp
```
A continuación se realiza una prueba la función del análisis con correos electrónicos de ejemplo.

```python
# Ahora procedemos a realizar pruebas de la función de análisis con correos
# electrónicos de ejemplos

import json
# Ejemplo de email phishing
phishing_email = """
From: security@amazonsupport.net
To: john.doe@company.com
Subject: Urgent: Your Amazon Account Has Been Locked

Dear Valued Customer,

We have detected unusual activity on your Amazon account. To prevent unauthorized access, we have temporarily locked your account for your protection.

To unlock your account and restore full access, please click on the link below and verify your information:

https://amaz0n-account-verify.com/unlock-account

If you do not verify your account within 24 hours, it will be permanently disabled.

Thank you for your prompt attention to this matter.

Best regards,
Amazon Security Team

This email was sent by Amazon.com. To ensure delivery to your inbox, please add security@amazonsupport.net to your address book.
"""

analysis = analyze_email(phishing_email)
print(json.dumps(analysis.model_dump(), indent=2))
```

En la siguiente imagen podemos evidenciar que la función empleada realiza el análisis del correo. Se puede observar que la probabilidad de phishing es alta y nos indica los elementos sospechos como la `dirección de correo del remitente` la `URL en el correo electrónico` y la `Sensación de urgencia`. Y nos detalla una razón de cada elemento sospechozo, también los detallas las acciones recomendadas para tomar en cuenta.

<img width="1484" alt="Captura de pantalla 2024-12-06 a la(s) 22 04 10" src="https://github.com/user-attachments/assets/636e3b2c-f8ed-47bd-b9be-9dde4525947a">

Se crea una interfaz gráfica de usuario sencilla para el análisis de correo electrónico mediante widgets de IPython. Solo aceptará como entrada un cuadro de texto, pero puede mejorarla utilizando un analizador de buzones de correo, compatibilidad con archivos adjuntos, extracciones de ioc, etc.

``` python
# Crear una interfaz gráfica de usuario para el análisis de correo electrónico

from IPython.display import display, HTML
import ipywidgets as widgets

# Componentes de la interfaz gráfica
email_input = widgets.Textarea(
    value='',
    placeholder='Paste the email content here...',
    description='Email:',
    disabled=False,
    layout={'width': '100%', 'height': '200px'}
)

analyze_button = widgets.Button(
    description='Analyze Email',
    disabled=False,
    button_style='primary',
    tooltip='Click to analyze the email',
    icon='check'
)

output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        print("Analyzing email...")
        try:
            analysis = analyze_email(email_input.value)
            display(HTML(format_analysis(analysis)))
        except Exception as e:
            print(f"An error occurred: {str(e)}")

analyze_button.on_click(on_button_clicked)

def format_analysis(analysis: SimplePhishingAnalysis) -> str:
    color = "red" if analysis.is_potential_phishing else "green"
    result = f"""
    <h2 style="color: {color};">{'Potential Phishing Detected' if analysis.is_potential_phishing else 'Likely Legitimate Email'}</h2>
    <p><strong>Phishing Probability:</strong> {analysis.phishing_probability.value}</p>
    <h3>Suspicious Elements:</h3>
    <ul>
    """
    for element in analysis.suspicious_elements:
        result += f"<li><strong>{element.element}:</strong> {element.reason}</li>"
    result += "</ul>"
    result += f"""
    <h3>Recommended Actions:</h3>
    <ul>
    """
    for action in analysis.recommended_actions:
        result += f"<li>{action}</li>"
    result += "</ul>"
    result += f"<h3>Explanation:</h3><p>{analysis.explanation}</p>"
    return result

# Se muestra la interfaz gráfica
display(email_input, analyze_button, output)
```




import requests
from typing import Callable
import os
import mlflow


# Constants and Variables
AUDIO_URL = 'https://www.nasa.gov/wp-content/uploads/2015/01/590325main_ringtone_kennedy_WeChoose.mp3'


def gather_audio() -> bytes:
    '''
    Acquire an audio file that is in the public domain

    Returns:
        (bytes) The downloaded audio.
    '''
    response = requests.get(AUDIO_URL)
    response.raise_for_status()
    audio = response.content

    return audio


def format_transcription(transcription: str) -> str:
    '''
    Function for formatting a long string by splitting into sentences and adding newlines.

    Args:
        transcription(str): The transcribed audio in string format.

    Returns:
        (str): Joined the sentences
    '''
    sentences = [
        sentence.strip() + ('.' if not sentence.endswith('.') else '')
        for sentence in transcription.split('. ')
        if sentence
    ]

    return ''.join(sentences)


def load_pretrained_model() -> Callable:
    '''
    Load the saved transcription pipeline from the most recent run
    
    Returns:
        (Callable): A python function wrapper with the model.
    '''
    mlflow.set_tracking_uri('http://localhost:5000')
    client = mlflow.tracking.MlflowClient()
    
    # Obtener el experimento
    experiment = client.get_experiment_by_name('LARA Whisper Transcription')
    if not experiment:
        raise Exception("No se encontró el experimento 'LARA Whisper Transcription'")
    
    # Obtener la ejecución más reciente
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise Exception("No se encontraron ejecuciones en el experimento")
    
    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/lara_whisper"
    print(f"Cargando modelo desde: {model_uri}")
    
    return mlflow.pyfunc.load_model(model_uri=model_uri)


def run_eval_flow(pyfunc_transcriber: Callable, audio: bytes):
    '''
    Execute the evaluation flow with MLflow tracking
    '''
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('LARA Whisper Transcription')

    with mlflow.start_run(run_name='Lara_Evaluation'):
        pyfunc_transcription = pyfunc_transcriber.predict([audio])
        mlflow.log_params({
            'Fisrt Entry Transcribed': format_transcription(pyfunc_transcription[0])
        })


if __name__ == '__main__':
    print("Iniciando evaluación del modelo Whisper...")
    
    # (1) Load Model (ahora directamente del run más reciente)
    print("Cargando el modelo entrenado más reciente...")
    pyfunc_transcriber = load_pretrained_model()
    
    # (2) Load audio
    print("Descargando audio de prueba...")
    audio = gather_audio()
    
    # (3) Predict
    print("Ejecutando evaluación...")
    run_eval_flow(pyfunc_transcriber, audio)
    
    print("Evaluación completada con éxito.")
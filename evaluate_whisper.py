import requests
from typing import Callable

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


def load_pretrained_model(model_uri: str) -> Callable:
    '''
    Load the saved transcription pipeline as a generic python function

    Args:
        model_uri(str): The uri of the pretrained model.

    Returns:
        (Callable): A python function wrapper with the model.
    '''
    return mlflow.pyfunc.load_model(model_uri=model_uri)


def run_eval_flow(pyfunc_transcriber: Callable, audio: bytes):
    '''
    '''
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('LARA Whisper Transcription')

    with mlflow.start_run(run_name='Lara'):
        pyfunc_transcription = pyfunc_transcriber.predict([audio])
        mlflow.log_params({
            'Fisrt Entry Transcribed': format_transcription(pyfunc_transcription[0])
        })


if __name__ == '__main__':

    # (1) Load Model
    pyfunc_transcriber = load_pretrained_model(model_uri='models:/WhisperLara/latest')

    # (2) Load audio
    audio = gather_audio()

    # (3) Predict
    run_eval_flow(pyfunc_transcriber, audio)
    
import requests
from typing import Any

import torch
import mlflow
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration, pipeline


# Constants and Variables
MODEL_NAME = 'openai/whisper-medium'
TASK = 'automatic-speech-recognition'
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


def load_model() -> pipeline:
    '''
    Load the components and necessary configuration for Whisper ASR from the Hugging Face Hub.

    Return:
        (pipeline): Hugging Face pipeline instance
    '''
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return pipeline(
        task=TASK,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device
    )


def start_flow(audio_transcription_pipeline: pipeline, audio: bytes, model_config: dict) -> Any:
    '''
    Start and Log the pipeline.

    Args:
        audio_transcription_pipeline(pipeline): Hugging Face pipeline instance.
        audio(bytes): The audio to be transcribed.
        model_config(dict): The config for the model inside MLFlow.

    Returns:
        (Any): The MLFlow run information.
    '''
    signature = mlflow.models.infer_signature(
        audio,
        mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
        params=model_config,
    )
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('LARA Whisper Transcription')

    with mlflow.start_run():
        model_info = mlflow.transformers.log_model(
            transformers_model=audio_transcription_pipeline,
            artifact_path='lara_whisper',
            signature=signature,
            input_example=audio,
            model_config=model_config,
            # save_pretrained=False
        )

        # Register Model
        mlflow.register_model(model_info.model_uri, 'WhisperLara')

    return model_info


if __name__ == '__main__':
    
    # (1) Load audio
    audio = gather_audio()

    # (2) Load Model and Create Pipeline
    audio_transcription_pipeline = load_model()

    # (3) Run Flow
    model_info = start_flow(
        audio_transcription_pipeline,
        audio,
        model_config={
            'chunk_length_s': 20,
            'stride_length_s': [5, 3],
        }
    )
    
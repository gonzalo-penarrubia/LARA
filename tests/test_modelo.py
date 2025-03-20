import unittest
from src.modelo import Modelo


class TestModelo(unittest.TestCase):

    def setUp(self):
        self.model_name = 'whisper'
        self.s3_url = 's3://lara'
        self.model_file_path = 'model/whisper/'
        self.data_file_path = 'data/audio.mp3'
        self.modelo = Modelo(self.model_name, self.s3_url, self.model_file_path, self.data_file_path)

    
    def test_init(self):
        self.assertEqual(self.modelo.model_name, self.model_name)
        self.assertEqual(self.modelo.s3_url, self.s3_url)
        self.assertEqual(self.modelo.model_file_path, self.model_file_path)
        self.assertEqual(self.modelo.data_file_path, self.data_file_path)
        self.assertEqual(self.modelo.model, "mocked_model")
        self.assertEqual(self.modelo.audio_data, "mocked_audio_data")

    
    def test_load_model(self):
        response = self.modelo.load_model()
        self.assertEqual(response, "mocked_model")

    
    def test_load_text(self):
        response = self.modelo.load_text()
        self.assertEqual(response, "mocked_audio_data")

    
    def test_transcribe(self):
        response = self.modelo.transcribe()
        self.assertEqual(response, "mocked transcribed text")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
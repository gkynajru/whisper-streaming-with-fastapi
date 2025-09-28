import pyaudio
from vosk import Model, KaldiRecognizer
import json

class RealTimeSTT:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        """
        Initialize the real-time speech recognition object.
        :param model_path: Path to the Vosk model directory.
        :param sample_rate: Sample rate (16000 Hz for most models).
        """
        self.model = Model(model_path)
        self.sample_rate = sample_rate
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

    def start_recognition(self):
        """
        Start real-time speech recognition from the microphone and print the results.
        """
        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Open audio stream from the microphone
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=4000)
        
        print("Starting speech recognition...")

        while True:
            # Read data from the microphone
            data = stream.read(4000)
            
            # If no data is available, break the loop
            if len(data) == 0:
                break
            
            # Process the audio data
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                result_json = json.loads(result)
                print("Recognition result:", result_json['text'])
            else:
                partial_result = self.recognizer.PartialResult()
                partial_result_json = json.loads(partial_result)
                print("Partial recognition:", partial_result_json['partial'])

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()


# Utility function to start speech recognition from the microphone
def start_microphone_recognition(model_type: str):
    model_paths = {
        "en": "./models/vosk/vosk-model-small-en-us-0.15.zip",  
        "vi": "D:/Project/benchmark-asr/models/vosk/vosk-model-vn-0.4"  
    }
    
    if model_type not in model_paths:
        raise ValueError("Only 'en' and 'vi' models are supported.")
    
    # Create the real-time speech recognition object
    stt = RealTimeSTT(model_paths[model_type])
    
    # Start recognition from the microphone
    stt.start_recognition()

if __name__ == "__main__":
    # Example usage: start recognition with the English model
    start_microphone_recognition("vi")

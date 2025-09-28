#!/usr/bin/env python3
"""
Streaming Whisper ONNX with microphone input
Processes audio in chunks and displays results in real-time
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import WhisperTokenizer, WhisperProcessor
import pyaudio
import warnings
warnings.filterwarnings('ignore')

os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

class StreamingWhisper:
    """Whisper model for streaming transcription"""
    
    def __init__(self, model_path: str = "models/whisper-tiny"):
        self.model_path = Path(model_path)
        
        # Load models
        print("Loading Whisper models...")
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.encoder = ort.InferenceSession(
            str(self.model_path / "encoder_model.onnx"), 
            sess_options, 
            providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(self.model_path / "decoder_model.onnx"), 
            sess_options, 
            providers=providers
        )
        
        # Load processor and tokenizer
        self.processor = WhisperProcessor.from_pretrained(str(self.model_path))
        self.tokenizer = WhisperTokenizer.from_pretrained(str(self.model_path))
        
        # Get special tokens
        self.sot_token = self.tokenizer.encode("<|startoftranscript|>", add_special_tokens=False)[0]
        self.en_token = self.tokenizer.encode("<|en|>", add_special_tokens=False)[0]
        self.transcribe_token = self.tokenizer.encode("<|transcribe|>", add_special_tokens=False)[0]
        self.notimestamps_token = self.tokenizer.encode("<|notimestamps|>", add_special_tokens=False)[0]
        
        self.sample_rate = 16000
        self.chunk_length = 30  # Process 30 seconds at a time
        
        # For tracking previous text to detect new words
        self.previous_text = ""
    
    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk"""
        if len(audio_chunk) < self.sample_rate:  # Less than 1 second
            return ""
        
        # Process audio to mel spectrogram
        inputs = self.processor(audio_chunk, sampling_rate=self.sample_rate, return_tensors="np")
        mel_features = inputs.input_features
        
        # Encode audio
        encoder_outputs = self.encoder.run(None, {"input_features": mel_features})[0]
        
        # Decode
        decoder_input_ids = np.array([[
            self.sot_token,
            self.en_token,
            self.transcribe_token,
            self.notimestamps_token
        ]], dtype=np.int64)
        
        generated_tokens = []
        max_length = 100
        
        for _ in range(max_length):
            outputs = self.decoder.run(None, {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_outputs
            })
            
            logits = outputs[0]
            next_token_id = np.argmax(logits[0, -1, :])
            
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # Check for repetition
            if len(generated_tokens) > 5:
                if all(t == next_token_id for t in generated_tokens[-5:]):
                    break
            
            generated_tokens.append(next_token_id)
            decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=-1)
        
        # Decode tokens
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text.strip()
    
    def get_new_words(self, full_text: str) -> str:
        """Get only the new words that weren't in previous text"""
        if not self.previous_text:
            self.previous_text = full_text
            return full_text
        
        # Find common prefix
        common_length = 0
        for i in range(min(len(self.previous_text), len(full_text))):
            if self.previous_text[i] == full_text[i]:
                common_length = i + 1
            else:
                break
        
        new_text = full_text[common_length:]
        self.previous_text = full_text
        return new_text

class MicrophoneStream:
    """Stream audio from microphone"""
    
    def __init__(self, rate=16000, chunk_duration=0.5):
        self.rate = rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
        # Find default input device
        self.device_index = 1
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print(f"Input Device id {i} - {self.p.get_device_info_by_host_api_device_index(0, i).get('name')}")
                if self.device_index is None:
                    self.device_index = i
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def start_stream(self):
        """Start the audio stream"""
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print(f"Microphone stream started (device {self.device_index})")
    
    def stop_stream(self):
        """Stop the audio stream"""
        self.stop_event.set()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("\nMicrophone stream stopped")
    
    def get_audio_chunk(self, duration=5.0):
        """Get audio chunk of specified duration"""
        frames = []
        chunks_needed = int(duration / self.chunk_duration)
        
        for _ in range(chunks_needed):
            if not self.audio_queue.empty():
                data = self.audio_queue.get()
                frames.append(np.frombuffer(data, dtype=np.float32))
        
        if frames:
            return np.concatenate(frames)
        return np.array([])

def streaming_transcribe(model_path="models/whisper-tiny", chunk_seconds=5):
    """Main streaming transcription with microphone"""
    
    print("="*60)
    print("WHISPER STREAMING TRANSCRIPTION")
    print("="*60)
    
    # Initialize model
    whisper = StreamingWhisper(model_path)
    print("Model loaded")
    
    # Initialize microphone
    mic = MicrophoneStream(rate=16000)
    mic.start_stream()
    
    print("\nTranscribing... (Press Ctrl+C to stop)")
    print("-"*60)
    
    # Buffer for accumulating audio
    audio_buffer = []
    buffer_duration = chunk_seconds
    last_process_time = time.time()
    full_transcript = ""
    
    try:
        while True:
            # Get audio chunk
            chunk = mic.get_audio_chunk(duration=0.5)
            
            if len(chunk) > 0:
                audio_buffer.append(chunk)
            
            # Process when we have enough audio
            current_time = time.time()
            if current_time - last_process_time >= buffer_duration and audio_buffer:
                # Combine buffer
                audio_data = np.concatenate(audio_buffer)
                
                # Keep last 2 seconds for context in next chunk
                keep_samples = int(2 * 16000)
                if len(audio_data) > keep_samples:
                    audio_buffer = [audio_data[-keep_samples:]]
                else:
                    audio_buffer = [audio_data]
                
                # Transcribe
                text = whisper.transcribe_chunk(audio_data)
                
                if text:
                    # For streaming effect, get only new words
                    new_words = whisper.get_new_words(text)
                    if new_words:
                        # Print new words in green
                        print(f"\033[92m{new_words}\033[0m", end=" ", flush=True)
                        full_transcript = text
                
                last_process_time = current_time
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n" + "-"*60)
        print("FINAL TRANSCRIPT:")
        print(full_transcript)
    finally:
        mic.stop_stream()

def test_with_file(audio_file, model_path="models/whisper-tiny"):
    """Test streaming simulation with an audio file"""
    import librosa
    
    print("="*60)
    print("SIMULATED STREAMING TEST")
    print("="*60)
    
    # Load audio
    audio, sr = librosa.load(audio_file, sr=16000)
    duration = len(audio) / sr
    print(f"Audio file: {audio_file}")
    print(f"Duration: {duration:.1f} seconds")
    print("-"*60)
    
    # Initialize model
    whisper = StreamingWhisper(model_path)
    
    # Process in chunks
    chunk_size = 3  # seconds
    chunk_samples = int(chunk_size * sr)
    overlap = int(1 * sr)  # 1 second overlap
    
    print("Streaming output:")
    print("-"*60)
    
    position = 0
    full_text = ""
    
    while position < len(audio):
        # Get chunk with overlap
        end = min(position + chunk_samples, len(audio))
        chunk = audio[position:end]
        
        # Transcribe chunk
        text = whisper.transcribe_chunk(chunk)
        
        if text:
            new_words = whisper.get_new_words(text)
            if new_words:
                print(new_words, end=" ", flush=True)
                time.sleep(0.5)  # Simulate real-time
                full_text = text
        
        # Move position (with overlap)
        position += (chunk_samples - overlap)
    
    print("\n" + "-"*60)
    print(f"Complete: {full_text}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Streaming Whisper transcription")
    parser.add_argument("--model", default="models/whisper-tiny", help="Path to model directory")
    parser.add_argument("--test-file", help="Test with audio file instead of microphone")
    parser.add_argument("--chunk-seconds", type=int, default=5, help="Seconds of audio to process at once")
    
    args = parser.parse_args()
    
    # Check if pyaudio is installed
    if not args.test_file:
        try:
            import pyaudio
        except ImportError:
            print("ERROR: PyAudio not installed. Install it with:")
            print("  pip install pyaudio")
            print("\nOr on Ubuntu:")
            print("  sudo apt-get install portaudio19-dev")
            print("  pip install pyaudio")
            sys.exit(1)
    
    if args.test_file:
        # Test with file
        test_with_file(args.test_file, args.model)
    else:
        # Live microphone streaming
        streaming_transcribe(args.model, args.chunk_seconds)
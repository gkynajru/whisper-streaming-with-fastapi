#!/usr/bin/env python3
"""
FastAPI server with WebSocket endpoint for real-time Whisper transcription
Handles streaming audio from browser and returns partial/final transcriptions
"""

import os
import sys
import json
import asyncio
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import warnings

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from transformers import WhisperTokenizer, WhisperProcessor

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# Import the StreamingWhisper class (you can also keep it in this file)
class StreamingWhisper:
    """Whisper model for streaming transcription"""
    
    def __init__(self, model_path: str = "models/whisper-tiny"):
        self.model_path = Path(model_path)
        
        # Load models
        print("Loading Whisper models...")
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use quantized models if available
        encoder_path = self.model_path / "encoder_model_quantized.onnx"
        decoder_path = self.model_path / "decoder_model_quantized.onnx"
        
        # Fall back to non-quantized if quantized not found
        if not encoder_path.exists():
            encoder_path = self.model_path / "encoder_model.onnx"
        if not decoder_path.exists():
            decoder_path = self.model_path / "decoder_model.onnx"
            
        self.encoder = ort.InferenceSession(
            str(encoder_path), 
            sess_options, 
            providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(decoder_path), 
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
        print("Model loaded successfully")
    
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


class AudioProcessor:
    """Handle audio buffering and processing for streaming"""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 3.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_buffer = []
        self.processed_audio = []
        self.overlap_samples = int(sample_rate * 1.0)  # 1 second overlap
        
    def add_audio(self, audio_bytes: bytes):
        """Add raw audio bytes to buffer"""
        # Convert bytes to int16 array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Convert to float32 and normalize
        audio_float = audio_array.astype(np.float32) / 32768.0
        self.audio_buffer.extend(audio_float)
        
    def get_chunk_for_processing(self) -> Optional[np.ndarray]:
        """Get audio chunk if enough data is available"""
        if len(self.audio_buffer) >= self.chunk_samples:
            # Get chunk for processing
            chunk = np.array(self.audio_buffer[:self.chunk_samples])
            
            # Keep overlap for context
            if len(self.audio_buffer) > self.chunk_samples:
                self.audio_buffer = self.audio_buffer[self.chunk_samples - self.overlap_samples:]
            else:
                self.audio_buffer = []
                
            return chunk
        return None
    
    def clear(self):
        """Clear all buffers"""
        self.audio_buffer = []
        self.processed_audio = []


class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.whisper_model: Optional[StreamingWhisper] = None
        self.transcription_states: Dict[str, str] = {}
        
    def initialize_model(self, model_path: str = "models/whisper-tiny"):
        """Initialize the Whisper model"""
        if not self.whisper_model:
            self.whisper_model = StreamingWhisper(model_path)
            
    async def connect(self, websocket: WebSocket, convo_id: str):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[convo_id] = websocket
        self.audio_processors[convo_id] = AudioProcessor()
        self.transcription_states[convo_id] = ""
        print(f"Client {convo_id} connected")
        
    def disconnect(self, convo_id: str):
        """Remove connection"""
        if convo_id in self.active_connections:
            del self.active_connections[convo_id]
        if convo_id in self.audio_processors:
            del self.audio_processors[convo_id]
        if convo_id in self.transcription_states:
            del self.transcription_states[convo_id]
        print(f"Client {convo_id} disconnected")
        
    async def send_transcription(self, convo_id: str, text: str, is_final: bool = False):
        """Send transcription result to client"""
        if convo_id in self.active_connections:
            message = {
                "content": text,
                "final": is_final,
                "timestamp": datetime.now().isoformat()
            }
            await self.active_connections[convo_id].send_json(message)
            
    async def process_audio(self, convo_id: str, audio_data: bytes):
        """Process incoming audio data"""
        if convo_id not in self.audio_processors:
            return
            
        processor = self.audio_processors[convo_id]
        processor.add_audio(audio_data)
        
        # Check if we have enough audio for processing
        chunk = processor.get_chunk_for_processing()
        if chunk is not None and self.whisper_model:
            # Run transcription
            try:
                transcription = self.whisper_model.transcribe_chunk(chunk)
                
                if transcription:
                    # Check if this is new text or update
                    previous_text = self.transcription_states.get(convo_id, "")
                    
                    if transcription != previous_text:
                        # Send partial result
                        await self.send_transcription(convo_id, transcription, is_final=False)
                        self.transcription_states[convo_id] = transcription
                        
                        # For demonstration, send final after a delay
                        # In production, you might want different logic
                        await asyncio.sleep(0.5)
                        await self.send_transcription(convo_id, transcription, is_final=True)
                        
            except Exception as e:
                print(f"Transcription error for {convo_id}: {e}")


# Initialize FastAPI app
app = FastAPI(title="Whisper Streaming API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connection manager
manager = ConnectionManager()

# Serve static files (for the HTML interface)
templates_path = Path("templates")
if templates_path.exists():
    app.mount("/static", StaticFiles(directory="templates"), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    manager.initialize_model("models/whisper-tiny")


@app.get("/")
async def get_index():
    """Serve the HTML interface"""
    html_path = Path("templates/index.html")
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Error: templates/index.html not found</h1>", 
            status_code=404
        )
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.websocket("/api/v1/agent/stt/{convo_id}")
async def websocket_endpoint(websocket: WebSocket, convo_id: str):
    """WebSocket endpoint for audio streaming and transcription"""
    await manager.connect(websocket, convo_id)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive()
            
            if "bytes" in data:
                # Binary audio data
                await manager.process_audio(convo_id, data["bytes"])
                
            elif "text" in data:
                # Text message (e.g., test message)
                message = json.loads(data["text"])
                print(f"Received text from {convo_id}: {message}")
                
                # Echo test messages
                if message.get("type") == "test":
                    await websocket.send_json({
                        "type": "test_response",
                        "message": "Test successful",
                        "timestamp": datetime.now().isoformat()
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(convo_id)
    except Exception as e:
        print(f"WebSocket error for {convo_id}: {e}")
        manager.disconnect(convo_id)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": manager.whisper_model is not None,
        "active_connections": len(manager.active_connections)
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3008,
        log_level="info"
    )
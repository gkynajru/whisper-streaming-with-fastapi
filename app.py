"""
    uvicorn app:app --host 0.0.0.0 --port 3008 --reload
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Logs the current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"{stage} - Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

class AudioBuffer:
    """Buffer for accumulating audio chunks with overlap"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.min_chunk_duration = 3.0
        self.max_chunk_duration = 30.0
        self.last_audio_time = None

    def add_audio(self, audio_bytes: bytes) -> None:
        """Add audio bytes to buffer"""
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer.extend(audio_data)
        self.last_audio_time = asyncio.get_event_loop().time()

    def get_duration(self) -> float:
        """Get current buffer duration in seconds"""
        return len(self.buffer) / self.sample_rate

    def has_enough_audio(self) -> bool:
        """Check if buffer has enough audio for transcription"""
        return self.get_duration() >= self.min_chunk_duration

    def should_flush(self, silence_timeout: float = 2.0) -> bool:
        """Check if buffer should be flushed due to silence"""
        if len(self.buffer) == 0 or self.last_audio_time is None:
            return False

        current_time = asyncio.get_event_loop().time()
        time_since_last_audio = current_time - self.last_audio_time

        return time_since_last_audio >= silence_timeout and self.get_duration() > 0.5

    def get_audio_chunk(self, force_flush: bool = False) -> Optional[np.ndarray]:
        """Get audio chunk for transcription"""
        if force_flush:
            if len(self.buffer) == 0:
                return None
            chunk = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            return chunk

        if not self.has_enough_audio():
            return None

        chunk_samples = int(min(self.get_duration(), self.max_chunk_duration) * self.sample_rate)
        chunk = np.array(self.buffer[:chunk_samples], dtype=np.float32)

        # Keep 2-second overlap
        overlap_samples = int(2.0 * self.sample_rate)
        if len(self.buffer) > chunk_samples:
            self.buffer = self.buffer[chunk_samples - overlap_samples:]
        else:
            self.buffer = []

        return chunk

    def clear(self) -> None:
        """Clear buffer"""
        self.buffer = []
        self.last_audio_time = None


class TranscriptAccumulator:
    """Handles text accumulation with overlap deduplication"""

    def __init__(self):
        self.full_transcript = ""
        self.previous_chunk_text = ""

    def add_chunk(self, chunk_text: str) -> str:
        """Add new chunk and return only new words"""
        if not chunk_text:
            return ""

        chunk_text = chunk_text.strip()

        if not self.previous_chunk_text:
            self.previous_chunk_text = chunk_text
            self.full_transcript = chunk_text
            return chunk_text

        new_words = self._extract_new_words(self.previous_chunk_text, chunk_text)
        self.previous_chunk_text = chunk_text

        if new_words:
            # Remove period before appending for continuous flow
            if self.full_transcript and self.full_transcript[-1] in '.。':
                self.full_transcript = self.full_transcript[:-1]

            if self.full_transcript and not self.full_transcript[-1].isspace():
                self.full_transcript += " "

            self.full_transcript += new_words

        return new_words

    def _extract_new_words(self, previous: str, current: str) -> str:
        """Extract new words from current text"""
        # Character-level matching
        common_length = 0
        min_len = min(len(previous), len(current))

        for i in range(min_len):
            if previous[i] == current[i]:
                common_length = i + 1
            else:
                break

        new_text = current[common_length:].strip()

        # Word-level fallback
        if common_length == 0 and previous:
            previous_words = previous.split()
            current_words = current.split()

            max_match = min(len(previous_words), len(current_words))
            match_count = 0

            for i in range(1, max_match + 1):
                if previous_words[-i:] == current_words[:i]:
                    match_count = i

            if match_count > 0:
                new_text = " ".join(current_words[match_count:])

        return new_text

    def get_full_transcript(self) -> str:
        """Get the complete accumulated transcript"""
        return self.full_transcript

    def clear(self) -> None:
        """Reset accumulator"""
        self.full_transcript = ""
        self.previous_chunk_text = ""


class LocalSTTSession:
    """Local Whisper STT Session"""

    # Class-level model cache
    _models: Dict[str, WhisperModel] = {}
    _model_paths = {
        "en": "models/fast_whisper/openai-whisper-base-ct2",
        "vi": "models/fast_whisper/vinai-phowhisper-base-ct2",
    }

    def __init__(self, websocket: WebSocket, language: Literal["en", "vi"] = "en"):
        self.ws = websocket
        self.language = language
        self.model = self._get_or_load_model(language)
        self.audio_buffer = AudioBuffer()
        self.transcript_accumulator = TranscriptAccumulator()
        self.shutdown_event = asyncio.Event()
        self.silence_timeout = 2.0
        self.client_timeout = 300  # 5 minutes

    @classmethod
    def _get_or_load_model(cls, language: Literal["en", "vi"]) -> WhisperModel:
        """Get cached model or load new one"""
        if language not in cls._models:
            model_path = cls._model_paths[language]
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    f"Please run model conversion first:\n"
                    f"ct2-transformers-converter --model "
                    f"{'openai/whisper-tiny' if language == 'en' else 'vinai/PhoWhisper-tiny'} "
                    f"--output_dir {model_path} --quantization int8"
                )

            log_memory_usage("Before loading model")
            logger.info(f"Loading {language.upper()} model from {model_path}...")
            cls._models[language] = WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            log_memory_usage("After loading model")
            logger.info(f"{language.upper()} model loaded")

        return cls._models[language]

    async def start(self) -> None:
        """Start STT session"""
        await self.ws.accept()
        logger.info(f"STT session started: language={self.language}")

        reader_task = asyncio.create_task(self.reader(), name="reader")
        processor_task = asyncio.create_task(self.processor(), name="processor")

        try:
            await asyncio.gather(reader_task, processor_task, return_exceptions=True)
        except Exception as e:
            logger.exception(f"Error in STT session: {e}")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown session and send final transcript"""
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()

            # Process remaining audio
            await self._flush_remaining_audio()

            # Send complete transcript
            full_text = self.transcript_accumulator.get_full_transcript()
            if full_text:
                try:
                    await self.ws.send_json({
                        "content": full_text,
                        "final": True,
                        "complete": True,
                        "timestamp": asyncio.get_event_loop().time()
                    })
                except Exception:
                    pass

            # Send end signal
            try:
                await self.ws.send_json({"content": "#END", "final": True})
            except Exception:
                pass

            try:
                await self.ws.close()
            except Exception:
                pass

            logger.info("STT session ended")

    async def reader(self) -> None:
        """Read audio data from WebSocket"""
        try:
            async with asyncio.timeout(self.client_timeout) as cm:
                while not self.shutdown_event.is_set():
                    data = await self.ws.receive_bytes()
                    if not data:
                        break

                    self.audio_buffer.add_audio(data)

                    # Reschedule timeout
                    new_deadline = asyncio.get_running_loop().time() + self.client_timeout
                    cm.reschedule(new_deadline)

        except asyncio.TimeoutError:
            logger.info("Client timeout - no audio received")
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as e:
            logger.exception(f"Error in reader: {e}")
        finally:
            if not self.shutdown_event.is_set():
                await self.shutdown()

    async def processor(self) -> None:
        """Process audio buffer and transcribe"""
        try:
            while not self.shutdown_event.is_set():
                await asyncio.sleep(0.5)

                # Check for normal chunk processing
                if self.audio_buffer.has_enough_audio():
                    audio_chunk = self.audio_buffer.get_audio_chunk(force_flush=False)
                    if audio_chunk is not None:
                        await self._process_chunk(audio_chunk)

                # Check for silence-based flush
                elif self.audio_buffer.should_flush(self.silence_timeout):
                    logger.info("Flushing buffer due to silence")
                    audio_chunk = self.audio_buffer.get_audio_chunk(force_flush=True)
                    if audio_chunk is not None:
                        await self._process_chunk(audio_chunk)

        except Exception as e:
            logger.exception(f"Error in processor: {e}")
        finally:
            if not self.shutdown_event.is_set():
                await self.shutdown()

    async def _process_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process a single audio chunk"""
        try:
            # Transcribe
            chunk_text = await asyncio.to_thread(self._transcribe, audio_chunk)

            if not chunk_text:
                return

            # Extract only new words
            new_words = self.transcript_accumulator.add_chunk(chunk_text)

            if new_words:
                # Send new words with full transcript
                await self.ws.send_json({
                    "content": new_words,
                    "final": False,
                    "full_transcript": self.transcript_accumulator.get_full_transcript(),
                    "timestamp": asyncio.get_event_loop().time()
                })

        except Exception as e:
            logger.exception(f"Error processing chunk: {e}")

    async def _flush_remaining_audio(self) -> None:
        """Process any remaining audio before shutdown"""
        if self.audio_buffer.get_duration() > 0.5:
            logger.info(f"Final flush: {self.audio_buffer.get_duration():.2f}s remaining")
            audio_chunk = self.audio_buffer.get_audio_chunk(force_flush=True)
            if audio_chunk is not None:
                chunk_text = await asyncio.to_thread(self._transcribe, audio_chunk)
                if chunk_text:
                    new_words = self.transcript_accumulator.add_chunk(chunk_text)
                    if new_words:
                        try:
                            await self.ws.send_json({
                                "content": new_words,
                                "final": True,
                                "full_transcript": self.transcript_accumulator.get_full_transcript(),
                                "timestamp": asyncio.get_event_loop().time()
                            })
                        except Exception:
                            pass

    def _transcribe(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk using faster-whisper"""
        try:
            segments, info = self.model.transcribe(
                audio_chunk,
                beam_size=5,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=1000,
                ),
            )

            text = " ".join([segment.text.strip() for segment in segments])
            return text.strip()

        except Exception as e:
            logger.exception(f"Transcription error: {e}")
            return ""


app = FastAPI(
    title="Local Whisper STT Streaming",
    description="Real-time speech-to-text using local Whisper models",
    version="2.0.0"
)


# Serve static files (templates)
templates_path = Path("templates")
if templates_path.exists():
    app.mount("/static", StaticFiles(directory="templates"), name="static")


@app.on_event("startup")
async def startup_event():
    """Warmup models on startup (optional)"""
    logger.info("Starting Local Whisper STT server...")
    logger.info(f"Templates directory: {templates_path.absolute()}")

    # Optional: Pre-load models for faster first request
    # Uncomment to enable:
    # try:
    #     logger.info("Warming up models...")
    #     LocalSTTSession._get_or_load_model("en")
    #     LocalSTTSession._get_or_load_model("vi")
    #     logger.info("Models warmed up")
    # except Exception as e:
    #     logger.warning(f"Model warmup failed: {e}")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the HTML interface"""
    html_path = templates_path / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>Error: templates/index.html not found</h1>",
            status_code=404
        )

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.websocket("/api/v1/agent/local-stt/{convo_id}/{language}")
async def local_stt_endpoint(
    websocket: WebSocket,
    convo_id: str,
    language: Literal["en", "vi"]
):
    """
    WebSocket endpoint for local Whisper STT

    Args:
        websocket: WebSocket connection
        convo_id: Conversation ID for tracking
        language: Language selection - "en" or "vi"
    """
    logger.info(f"New connection: convo_id={convo_id}, language={language}")

    try:
        session = LocalSTTSession(websocket, language=language)
        await session.start()
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        await websocket.close(code=1011, reason=str(e))
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    loaded_models = list(LocalSTTSession._models.keys())
    return {
        "status": "healthy",
        "models_loaded": loaded_models,
        "available_languages": ["en", "vi"]
    }


@app.get("/models")
async def list_models():
    """List available models"""
    models = {}
    for lang, path in LocalSTTSession._model_paths.items():
        models[lang] = {
            "path": path,
            "exists": Path(path).exists(),
            "loaded": lang in LocalSTTSession._models
        }
    return models


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3008,
        log_level="info"
    )

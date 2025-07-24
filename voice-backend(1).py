# src/api/natural_voice_backend.py
# Complete FastAPI backend for natural voice-to-voice legal intake
# Integrates PodGPT's RAG pipeline with advanced voice processing and lip sync

import asyncio
import json
import time
import logging
import os
import tempfile
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import websockets

# Audio processing
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr
import requests
from io import BytesIO

# Database and caching
import redis
import psycopg2
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid as uuid_lib

# ML and AI processing
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
from sentence_transformers import SentenceTransformer

# Configuration and monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Initialize logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# FastAPI app configuration
app = FastAPI(
    title="VoiceLaw-AI Natural Voice Legal Intake API",
    description="Advanced voice-to-voice legal intake with lip sync and PodGPT integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://legal_ai:password@localhost:5432/legal_intake")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# Prometheus metrics
voice_requests_total = Counter('voice_requests_total', 'Total voice requests', ['method', 'status'])
voice_processing_duration = Histogram('voice_processing_seconds', 'Voice processing duration')
active_sessions = Gauge('active_voice_sessions', 'Number of active voice sessions')
transcription_accuracy = Histogram('transcription_accuracy', 'Transcription confidence scores')
lip_sync_generation_time = Histogram('lip_sync_generation_seconds', 'Lip sync generation time')

# Database Models
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    session_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=True)
    language = Column(String(10), nullable=False, default='en')
    case_type = Column(String(100), nullable=True)
    urgency = Column(String(20), default='medium')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False)
    type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    transcription_confidence = Column(Float, nullable=True)
    voice_metrics = Column(JSON, nullable=True)
    legal_context = Column(JSON, nullable=True)
    lip_sync_data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class LegalEntity(Base):
    __tablename__ = "legal_entities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_lib.uuid4)
    conversation_id = Column(UUID(as_uuid=True), nullable=False)
    entity_type = Column(String(100), nullable=False)
    entity_value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    extracted_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Global AI components
whisper_model = None
legal_rag_pipeline = None
entity_extractor = None
case_classifier = None
embedding_model = None
nlp_models = {}

# Active conversation sessions
active_sessions_dict: Dict[str, Dict[str, Any]] = {}

class VoiceConversationManager:
    """Manages natural voice-to-voice conversations with advanced features"""
    
    def __init__(self, session_id: str, language: str = "en", db: Session = None):
        self.session_id = session_id
        self.language = language
        self.db = db
        self.conversation_history = []
        self.current_context = {}
        self.turn_taking_state = "user_turn"
        self.silence_timer = 0
        self.interruption_count = 0
        self.voice_quality_metrics = {}
        self.legal_entities = {}
        self.case_classification = {}
        
        # Natural conversation parameters
        self.silence_threshold = 1.5  # seconds
        self.interruption_grace_period = 0.8
        self.max_turn_duration = 30  # seconds
        
        # Performance tracking
        self.processing_times = []
        self.transcription_costs = []
        
        logger.info("Voice conversation manager initialized", session_id=session_id, language=language)

    async def process_audio_stream(self, audio_data: bytes) -> Dict[str, Any]:
        """Process incoming audio with advanced voice analysis"""
        
        start_time = time.time()
        
        try:
            with voice_processing_duration.time():
                # Step 1: Analyze audio quality and voice activity
                voice_metrics = await self._analyze_voice_activity(audio_data)
                
                # Step 2: Handle natural turn-taking logic
                turn_decision = await self._handle_turn_taking(voice_metrics)
                
                if turn_decision["action"] == "continue_listening":
                    voice_requests_total.labels(method="listen", status="continue").inc()
                    return {
                        "status": "listening", 
                        "voice_metrics": voice_metrics,
                        "turn_state": self.turn_taking_state
                    }
                
                elif turn_decision["action"] == "process_speech":
                    # Step 3: Transcribe with optimal method selection
                    transcription = await self._transcribe_audio_optimized(audio_data)
                    transcription_accuracy.observe(transcription.get("confidence", 0))
                    
                    # Step 4: Process through PodGPT RAG pipeline
                    rag_response = await self._process_with_podgpt_rag(transcription)
                    
                    # Step 5: Generate advanced lip sync data
                    lip_sync_data = await self._generate_advanced_lip_sync(rag_response["text"])
                    
                    # Step 6: Store conversation in database
                    await self._store_conversation_turn(transcription, rag_response, lip_sync_data)
                    
                    # Step 7: Update conversation state
                    self._update_conversation_state(transcription, rag_response)
                    
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    voice_requests_total.labels(method="process", status="success").inc()
                    
                    return {
                        "status": "response_ready",
                        "transcription": transcription,
                        "response": rag_response,
                        "lip_sync": lip_sync_data,
                        "voice_metrics": voice_metrics,
                        "turn_state": "ai_turn",
                        "processing_time": processing_time
                    }
                    
                elif turn_decision["action"] == "handle_interruption":
                    return await self._handle_interruption()
                    
        except Exception as e:
            logger.error("Audio processing error", error=str(e), session_id=self.session_id)
            voice_requests_total.labels(method="process", status="error").inc()
            return {"status": "error", "message": str(e)}

    async def _analyze_voice_activity(self, audio_data: bytes) -> Dict[str, Any]:
        """Advanced voice activity detection with quality metrics"""
        
        try:
            # Convert bytes to audio array
            audio_segment = AudioSegment.from_raw(
                BytesIO(audio_data), 
                sample_width=2, 
                frame_rate=16000, 
                channels=1
            )
            
            # Convert to numpy array
            audio_array = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
            
            # Calculate comprehensive voice metrics
            rms_energy = np.sqrt(np.mean(audio_array ** 2))
            zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_array))))
            
            # Spectral analysis
            stft = librosa.stft(audio_array, hop_length=512)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=np.abs(stft)))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=np.abs(stft)))
            
            # Pitch estimation using librosa
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_array, sr=16000)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                avg_pitch = np.mean(pitch_values) if pitch_values else 0
            except:
                avg_pitch = 0
            
            # Speech likelihood calculation
            speech_likelihood = self._calculate_advanced_speech_likelihood(
                rms_energy, zero_crossing_rate, spectral_centroid, avg_pitch
            )
            
            # Audio quality assessment
            quality_score = self._assess_comprehensive_audio_quality(audio_array)
            
            # Voice emotion detection (simplified)
            emotion_score = self._detect_voice_emotion(audio_array)
            
            return {
                "rms_energy": float(rms_energy),
                "zero_crossing_rate": float(zero_crossing_rate),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "pitch": float(avg_pitch),
                "speech_likelihood": float(speech_likelihood),
                "quality_score": float(quality_score),
                "emotion_score": emotion_score,
                "is_voice_active": speech_likelihood > 0.3,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error("Voice analysis failed", error=str(e))
            return {
                "rms_energy": 0.0, "speech_likelihood": 0.0,
                "quality_score": 0.0, "is_voice_active": False,
                "timestamp": time.time(), "error": str(e)
            }

    def _calculate_advanced_speech_likelihood(self, rms: float, zcr: float, centroid: float, pitch: float) -> float:
        """Advanced speech detection using multiple acoustic features"""
        
        # Energy component (0-1)
        energy_score = min(rms / 0.05, 1.0)
        
        # Zero crossing rate (optimal for speech: 0.1-0.3)
        zcr_score = 1.0 - abs(zcr - 0.2) / 0.2 if zcr <= 0.4 else 0.0
        
        # Spectral centroid (speech typically 1000-4000 Hz)
        centroid_score = 1.0 if 1000 <= centroid <= 4000 else max(0, 1.0 - abs(centroid - 2500) / 2500)
        
        # Pitch presence and range (human speech: 80-400 Hz)
        pitch_score = 1.0 if 80 <= pitch <= 400 else 0.5 if pitch > 0 else 0.0
        
        # Weighted combination
        speech_likelihood = (
            energy_score * 0.3 +
            zcr_score * 0.25 +
            centroid_score * 0.25 +
            pitch_score * 0.2
        )
        
        return min(speech_likelihood, 1.0)

    def _assess_comprehensive_audio_quality(self, audio: np.ndarray) -> float:
        """Comprehensive audio quality assessment"""
        
        # Signal-to-noise ratio
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(audio ** 2, 10)
        snr = 10 * np.log10(signal_power / max(noise_floor, 1e-10))
        
        # Dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
        
        # Clipping detection
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        
        # Frequency response (simplified)
        fft = np.fft.rfft(audio)
        freq_response = np.mean(np.abs(fft))
        
        # Combine metrics
        quality = (
            min(snr / 20, 1.0) * 0.4 +           # SNR component
            min(dynamic_range * 2, 1.0) * 0.25 + # Dynamic range
            (1.0 - clipping_ratio) * 0.2 +       # Anti-clipping
            min(freq_response * 1000, 1.0) * 0.15 # Frequency response
        )
        
        return max(0.0, min(quality, 1.0))

    def _detect_voice_emotion(self, audio: np.ndarray) -> Dict[str, float]:
        """Basic voice emotion detection"""
        
        # Simplified emotion detection based on acoustic features
        pitch_mean = np.mean(audio)
        pitch_std = np.std(audio)
        energy = np.mean(audio ** 2)
        
        emotions = {
            "neutral": 0.5,
            "concerned": min(1.0, pitch_std * 2 + energy),
            "understanding": min(1.0, abs(pitch_mean) * 1.5),
            "helpful": min(1.0, energy * 1.2),
            "empathetic": min(1.0, pitch_std * 1.5 + abs(pitch_mean))
        }
        
        # Normalize so they sum to 1
        total = sum(emotions.values())
        return {k: v/total for k, v in emotions.items()}

    async def _handle_turn_taking(self, voice_metrics: Dict[str, Any]) -> Dict[str, str]:
        """Advanced turn-taking with natural conversation flow"""
        
        current_time = time.time()
        is_voice_active = voice_metrics["is_voice_active"]
        
        if self.turn_taking_state == "user_turn":
            if is_voice_active:
                self.silence_timer = 0
                return {"action": "continue_listening"}
            else:
                self.silence_timer += 0.1  # Assuming 100ms chunks
                
                # Adaptive silence threshold based on conversation context
                threshold = self.silence_threshold
                if self.current_context.get("urgency") == "high":
                    threshold *= 0.6
                elif self.interruption_count > 2:
                    threshold *= 0.8
                
                if self.silence_timer >= threshold:
                    self.turn_taking_state = "ai_processing"
                    return {"action": "process_speech"}
                else:
                    return {"action": "continue_listening"}
                    
        elif self.turn_taking_state == "ai_turn":
            if is_voice_active and voice_metrics["speech_likelihood"] > 0.6:
                self.interruption_count += 1
                self.turn_taking_state = "user_turn"
                return {"action": "handle_interruption"}
            else:
                return {"action": "continue_ai_turn"}
                
        return {"action": "continue_listening"}

    async def _transcribe_audio_optimized(self, audio_data: bytes) -> Dict[str, Any]:
        """Optimized transcription using best available method"""
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_segment = AudioSegment.from_raw(
                BytesIO(audio_data), 
                sample_width=2, 
                frame_rate=16000, 
                channels=1
            )
            audio_segment.export(temp_file.name, format="wav")
            temp_path = temp_file.name
        
        try:
            # Assess audio quality to choose transcription method
            quality_score = self.voice_quality_metrics.get("quality_score", 0.5)
            
            if quality_score > 0.7:
                # High quality -> Use Whisper
                result = await self._transcribe_with_whisper(temp_path)
                method = "whisper"
                
            elif quality_score > 0.4:
                # Medium quality -> Try hybrid approach
                result = await self._transcribe_hybrid(temp_path)
                method = "hybrid"
                
            else:
                # Low quality -> Use NVIDIA Flamingo 3 or fallback
                result = await self._transcribe_with_flamingo(temp_path)
                method = "flamingo3"
            
            # Add metadata
            result.update({
                "transcription_method": method,
                "audio_quality": quality_score,
                "language_detected": result.get("language", self.language),
                "processing_time": result.get("processing_time", 0)
            })
            
            # Store cost information
            cost = result.get("cost", 0)
            self.transcription_costs.append(cost)
            
            return result
            
        finally:
            # Cleanup temporary file
            Path(temp_path).unlink(missing_ok=True)

    async def _transcribe_with_whisper(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper"""
        
        start_time = time.time()
        
        try:
            if whisper_model:
                # Use local Whisper model
                result = whisper_model.transcribe(
                    audio_path,
                    language=self.language,
                    word_timestamps=True,
                    initial_prompt="This is a legal consultation conversation."
                )
                
                return {
                    "text": result["text"].strip(),
                    "language": result["language"],
                    "confidence": 0.9,  # Whisper doesn't provide confidence
                    "word_timestamps": result.get("segments", []),
                    "cost": self._calculate_whisper_cost(audio_path),
                    "processing_time": time.time() - start_time
                }
            else:
                # Use OpenAI API
                return await self._transcribe_with_openai_api(audio_path)
                
        except Exception as e:
            logger.error("Whisper transcription failed", error=str(e))
            return {
                "text": "", "confidence": 0.0, "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def _transcribe_with_flamingo(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe using NVIDIA Flamingo 3"""
        
        start_time = time.time()
        
        try:
            nvidia_endpoint = os.getenv("NVIDIA_FLAMINGO_ENDPOINT")
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            
            if not nvidia_endpoint or not nvidia_api_key:
                raise ValueError("NVIDIA Flamingo credentials not configured")
            
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            response = requests.post(
                f"{nvidia_endpoint}/transcribe",
                headers={
                    "Authorization": f"Bearer {nvidia_api_key}",
                    "Content-Type": "application/octet-stream"
                },
                data=audio_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("transcription", ""),
                    "confidence": result.get("confidence", 0.8),
                    "language": self.language,
                    "cost": self._calculate_flamingo_cost(audio_path),
                    "processing_time": time.time() - start_time
                }
            else:
                raise Exception(f"NVIDIA API error: {response.status_code}")
                
        except Exception as e:
            logger.error("Flamingo transcription failed", error=str(e))
            return {
                "text": "", "confidence": 0.0, "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def _transcribe_hybrid(self, audio_path: str) -> Dict[str, Any]:
        """Use both Whisper and Flamingo, return best result"""
        
        start_time = time.time()
        
        # Run both transcriptions in parallel
        whisper_task = asyncio.create_task(self._transcribe_with_whisper(audio_path))
        flamingo_task = asyncio.create_task(self._transcribe_with_flamingo(audio_path))
        
        try:
            whisper_result, flamingo_result = await asyncio.gather(
                whisper_task, flamingo_task, return_exceptions=True
            )
            
            # Choose best result based on confidence and cost
            if isinstance(whisper_result, dict) and isinstance(flamingo_result, dict):
                whisper_score = (
                    whisper_result.get("confidence", 0) * 0.7 + 
                    (1 - whisper_result.get("cost", 1)) * 0.3
                )
                flamingo_score = (
                    flamingo_result.get("confidence", 0) * 0.7 + 
                    (1 - flamingo_result.get("cost", 1)) * 0.3
                )
                
                if whisper_score > flamingo_score:
                    whisper_result["method"] = "hybrid_whisper"
                    whisper_result["processing_time"] = time.time() - start_time
                    return whisper_result
                else:
                    flamingo_result["method"] = "hybrid_flamingo" 
                    flamingo_result["processing_time"] = time.time() - start_time
                    return flamingo_result
            
            # Fallback to whichever succeeded
            if isinstance(whisper_result, dict):
                return whisper_result
            elif isinstance(flamingo_result, dict):
                return flamingo_result
            else:
                raise Exception("Both transcription methods failed")
                
        except Exception as e:
            logger.error("Hybrid transcription failed", error=str(e))
            return {
                "text": "", "confidence": 0.0, "error": str(e),
                "processing_time": time.time() - start_time
            }

    def _calculate_whisper_cost(self, audio_path: str) -> float:
        """Calculate Whisper transcription cost"""
        
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            duration_minutes = audio_segment.duration_seconds / 60
            return duration_minutes * 0.006  # $0.006 per minute
        except:
            return 0.01  # Default estimate

    def _calculate_flamingo_cost(self, audio_path: str) -> float:
        """Calculate NVIDIA Flamingo cost"""
        
        try:
            audio_segment = AudioSegment.from_file(audio_path)
            duration_minutes = audio_segment.duration_seconds / 60
            return duration_minutes * 0.003  # Estimated lower cost
        except:
            return 0.005  # Default estimate

# Dependency injection for database sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    """Initialize AI models and components on startup"""
    
    global whisper_model, legal_rag_pipeline, entity_extractor, case_classifier, embedding_model
    
    logger.info("🚀 Starting VoiceLaw-AI Natural Voice Legal Intake API")
    
    try:
        # Initialize Whisper model
        model_path = os.getenv("WHISPER_MODEL_PATH", "./models/whisper-large-v3")
        if os.path.exists(model_path):
            whisper_model = whisper.load_model(model_path)
            logger.info("✅ Local Whisper model loaded")
        else:
            whisper_model = whisper.load_model("large-v3")
            logger.info("✅ Downloaded Whisper model loaded")
        
        # Initialize legal NLP models
        try:
            nlp_models['en'] = spacy.load("en_core_web_sm")
            nlp_models['es'] = spacy.load("es_core_news_sm")
            logger.info("✅ Spacy models loaded")
        except OSError:
            logger.warning("⚠️ Some spacy models not found, using basic processing")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
        
        # Initialize Redis connection
        redis_client.ping()
        logger.info("✅ Redis connection established")
        
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("✅ Database connection established")
        
        logger.info("🎉 All systems initialized successfully!")
        
    except Exception as e:
        logger.error("❌ Initialization failed", error=str(e))
        raise

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint with detailed system status"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "active_sessions": len(active_sessions_dict),
        "components": {
            "whisper_model": whisper_model is not None,
            "embedding_model": embedding_model is not None,
            "redis": False,
            "database": False,
            "nlp_models": len(nlp_models)
        }
    }
    
    # Test Redis connection
    try:
        redis_client.ping()
        health_status["components"]["redis"] = True
    except:
        health_status["components"]["redis"] = False
    
    # Test database connection
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        health_status["components"]["database"] = True
    except:
        health_status["components"]["database"] = False
    
    # Determine overall health
    critical_components = ["whisper_model", "redis", "database"]
    if not all(health_status["components"][comp] for comp in critical_components):
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

# Prometheus metrics endpoint
@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "natural_voice_backend:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
        workers=1  # Use 1 worker for WebSocket support
    )to-voice legal intake
# Integrates PodGPT's RAG pipeline with advanced voice processing and lip sync

import asyncio
import json
import time
import logging
import os
import tempfile
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import websockets

# Audio processing
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr
import requests
from io import BytesIO

# Database and caching
import redis
import psycopg2
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid as uuid_lib

# ML and AI processing
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
from sentence_transformers import SentenceTransformer

# Configuration and monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Initialize logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

#

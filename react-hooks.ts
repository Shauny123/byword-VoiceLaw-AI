// src/hooks/usePodGPTRag.ts
// React hooks for seamless PodGPT RAG integration with voice processing
// Connects frontend components to the natural voice backend

import { useState, useCallback, useRef, useEffect } from 'react';

// Types for PodGPT RAG integration
interface PodGPTRagOptions {
  language: string;
  legalDomain: string;
  apiEndpoint: string;
  wsEndpoint: string;
  onResponseGenerated: (response: string, context: any) => void;
  onThinkingStateChange: (thinking: boolean) => void;
  onErrorOccurred?: (error: string) => void;
  enableRetry?: boolean;
  maxRetries?: number;
}

interface RAGContext {
  retrievedDocuments: Array<{
    content: string;
    source: string;
    relevance: number;
    legalDomain: string;
  }>;
  legalContext: string[];
  urgency: 'low' | 'medium' | 'high';
  confidence: number;
  processingTime: number;
  entities: Record<string, any>;
  caseType: string;
}

interface RAGResponse {
  text: string;
  emotion: string;
  confidence: number;
  legalContext: string[];
  urgency: string;
  entities: Record<string, any>;
  caseType: string;
  ragSources: any[];
  processingMetadata: Record<string, any>;
}

interface VoiceProcessingOptions {
  language: string;
  enableQualityAnalysis: boolean;
  enableCostOptimization: boolean;
  onTranscriptionComplete: (transcript: string, confidence: number) => void;
  onAudioQualityChange: (quality: number) => void;
  onProcessingStateChange: (state: 'idle' | 'listening' | 'processing' | 'speaking') => void;
}

interface VoiceMetrics {
  volume: number;
  pitch: number;
  speechLikeness: number;
  quality: number;
  naturalness: number;
  emotion: Record<string, number>;
}

// Main hook for PodGPT RAG integration
export const usePodGPTRag = (options: PodGPTRagOptions) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastContext, setLastContext] = useState<RAGContext | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [retryCount, setRetryCount] = useState(0);
  
  const abortControllerRef = useRef<AbortController | null>(null);
  const processingQueueRef = useRef<Array<() => Promise<void>>>([]);
  const isProcessingQueueRef = useRef(false);

  const processUserInput = useCallback(async (
    userInput: string, 
    conversationHistory: any[]
  ) => {
    // Cancel any ongoing processing
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    abortControllerRef.current = new AbortController();
    
    // Add to processing queue for sequential processing
    return new Promise<void>((resolve, reject) => {
      processingQueueRef.current.push(async () => {
        try {
          setIsProcessing(true);
          options.onThinkingStateChange(true);
          
          const startTime = Date.now();
          
          // Step 1: Analyze legal content
          const legalAnalysis = await analyzeLegalContent(userInput, options.language);
          
          // Step 2: Retrieve relevant documents
          const retrievedDocs = await retrieveRelevantDocuments(
            userInput, 
            options.legalDomain,
            legalAnalysis,
            abortControllerRef.current.signal
          );
          
          // Step 3: Generate contextual response
          const response = await generateRAGResponse(
            userInput,
            retrievedDocs,
            conversationHistory,
            legalAnalysis,
            options.language,
            abortControllerRef.current.signal
          );
          
          const processingTime = Date.now() - startTime;
          
          // Build context for frontend
          const ragContext: RAGContext = {
            retrievedDocuments: retrievedDocs,
            legalContext: legalAnalysis.domains || [],
            urgency: legalAnalysis.urgency || 'medium',
            confidence: response.confidence,
            processingTime,
            entities: legalAnalysis.entities || {},
            caseType: legalAnalysis.case_type || 'general'
          };
          
          setLastContext(ragContext);
          setRetryCount(0); // Reset retry count on success
          options.onResponseGenerated(response.text, ragContext);
          
          resolve();
          
        } catch (error) {
          if (error.name !== 'AbortError') {
            console.error('RAG processing failed:', error);
            
            // Handle retry logic
            if (options.enableRetry && retryCount < (options.maxRetries || 3)) {
              setRetryCount(prev => prev + 1);
              setTimeout(() => {
                processingQueueRef.current.unshift(async () => {
                  await processUserInput(userInput, conversationHistory);
                });
                processQueue();
              }, 1000 * Math.pow(2, retryCount)); // Exponential backoff
            } else {
              options.onErrorOccurred?.(error.message);
              options.onResponseGenerated(
                getErrorResponse(options.language, error.message), 
                { urgency: 'low', legalContext: [], confidence: 0 } as RAGContext
              );
            }
            
            reject(error);
          }
        } finally {
          setIsProcessing(false);
          options.onThinkingStateChange(false);
        }
      });
      
      processQueue();
    });
  }, [options, retryCount]);

  const processQueue = useCallback(async () => {
    if (isProcessingQueueRef.current || processingQueueRef.current.length === 0) {
      return;
    }
    
    isProcessingQueueRef.current = true;
    
    while (processingQueueRef.current.length > 0) {
      const nextTask = processingQueueRef.current.shift();
      if (nextTask) {
        try {
          await nextTask();
        } catch (error) {
          console.error('Queue processing error:', error);
        }
      }
    }
    
    isProcessingQueueRef.current = false;
  }, []);

  const analyzeLegalContent = async (text: string, language: string) => {
    const response = await fetch(`${options.apiEndpoint}/api/legal-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
      signal: abortControllerRef.current?.signal
    });
    
    if (!response.ok) {
      throw new Error(`Legal analysis failed: ${response.statusText}`);
    }
    
    return await response.json();
  };

  const retrieveRelevantDocuments = async (
    query: string,
    domain: string, 
    analysis: any,
    signal: AbortSignal
  ) => {
    const response = await fetch(`${options.apiEndpoint}/api/rag/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        domain,
        analysis,
        max_docs: 5
      }),
      signal
    });
    
    if (!response.ok) {
      throw new Error(`Document retrieval failed: ${response.statusText}`);
    }
    
    return await response.json();
  };

  const generateRAGResponse = async (
    userInput: string,
    retrievedDocs: any[],
    history: any[],
    analysis: any,
    language: string,
    signal: AbortSignal
  ) => {
    const response = await fetch(`${options.apiEndpoint}/api/rag/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_input: userInput,
        retrieved_documents: retrievedDocs,
        conversation_history: history,
        legal_analysis: analysis,
        language,
        response_style: 'compassionate_legal'
      }),
      signal
    });
    
    if (!response.ok) {
      throw new Error(`Response generation failed: ${response.statusText}`);
    }
    
    return await response.json();
  };

  const getErrorResponse = (language: string, errorMessage: string): string => {
    const errorResponses: Record<string, string> = {
      en: "I apologize, but I encountered an issue processing your request. Could you please try rephrasing your question?",
      es: "Me disculpo, pero encontré un problema procesando su solicitud. ¿Podría reformular su pregunta?",
      fr: "Je m'excuse, mais j'ai rencontré un problème en traitant votre demande. Pourriez-vous reformuler votre question?",
      zh: "抱歉，我在处理您的请求时遇到了问题。您能重新表述一下您的问题吗？",
      ar: "أعتذر، لكنني واجهت مشكلة في معالجة طلبك. هل يمكنك إعادة صياغة سؤالك؟"
    };
    return errorResponses[language] || errorResponses.en;
  };

  const cancelProcessing = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Clear processing queue
    processingQueueRef.current = [];
    isProcessingQueueRef.current = false;
    
    setIsProcessing(false);
    options.onThinkingStateChange(false);
  }, [options]);

  const getProcessingStats = useCallback(() => {
    return {
      isProcessing,
      queueLength: processingQueueRef.current.length,
      retryCount,
      lastContext,
      connectionStatus
    };
  }, [isProcessing, retryCount, lastContext, connectionStatus]);

  return {
    processUserInput,
    isProcessing,
    lastContext,
    connectionStatus,
    retryCount,
    cancelProcessing,
    getProcessingStats
  };
};

// Hook for voice processing with WebSocket integration
export const useVoiceProcessing = (options: VoiceProcessingOptions) => {
  const [isListening, setIsListening] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [voiceMetrics, setVoiceMetrics] = useState<VoiceMetrics>({
    volume: 0,
    pitch: 0,
    speechLikeness: 0,
    quality: 0,
    naturalness: 0,
    emotion: {}
  });
  const [error, setError] = useState<string>('');
  const [transcriptionMethod, setTranscriptionMethod] = useState<'whisper' | 'flamingo3' | 'hybrid'>('whisper');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const processingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100, // High quality for better analysis
        }
      });

      // Set up audio analysis
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 2048;

      // Start recording
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await processAudioBlob(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsListening(true);
      setError('');
      options.onProcessingStateChange('listening');

      // Start audio level monitoring
      monitorAudioLevel();

    } catch (error) {
      console.error('Failed to start listening:', error);
      setError('Microphone access denied. Please allow microphone access.');
      options.onProcessingStateChange('idle');
    }
  }, [options]);

  const stopListening = useCallback(() => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
      setIsProcessingAudio(true);
      options.onProcessingStateChange('processing');

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    }
  }, [isListening, options]);

  const monitorAudioLevel = useCallback(() => {
    if (!analyserRef.current) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const frequencyData = new Uint8Array(bufferLength);

    const updateAudioLevel = () => {
      if (analyserRef.current && isListening) {
        analyserRef.current.getByteTimeDomainData(dataArray);
        analyserRef.current.getByteFrequencyData(frequencyData);
        
        // Calculate comprehensive voice metrics
        const volume = calculateRMSVolume(dataArray);
        const pitch = estimatePitch(dataArray);
        const speechLikeness = analyzeSpeechLikeness(frequencyData);
        const quality = assessAudioQuality(dataArray, frequencyData);
        const emotion = detectVoiceEmotion(dataArray, frequencyData);
        
        const metrics: VoiceMetrics = {
          volume,
          pitch,
          speechLikeness,
          quality,
          naturalness: Math.min(speechLikeness * quality, 1.0),
          emotion
        };
        
        setAudioLevel(volume);
        setVoiceMetrics(metrics);
        options.onAudioQualityChange(quality);
        
        // Determine optimal transcription method based on quality
        if (options.enableCostOptimization) {
          const optimalMethod = determineOptimalTranscriptionMethod(metrics);
          setTranscriptionMethod(optimalMethod);
        }
        
        animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
      }
    };

    updateAudioLevel();
  }, [isListening, options]);

  const processAudioBlob = useCallback(async (audioBlob: Blob) => {
    try {
      setIsProcessingAudio(true);
      
      // Analyze audio quality first if enabled
      let qualityAnalysis = null;
      if (options.enableQualityAnalysis) {
        qualityAnalysis = await analyzeAudioQuality(audioBlob);
      }
      
      // Choose transcription method
      const method = options.enableCostOptimization ? 
        (qualityAnalysis?.recommended_method || transcriptionMethod) : 
        'whisper';
      
      // Transcribe audio
      const transcription = await transcribeAudio(audioBlob, method, options.language);
      
      // Update processing stats
      if (qualityAnalysis) {
        console.log('Audio quality analysis:', qualityAnalysis);
        console.log('Transcription method used:', method);
        console.log('Estimated cost:', transcription.cost);
      }
      
      // Call completion callback
      options.onTranscriptionComplete(transcription.text, transcription.confidence);
      
    } catch (error) {
      console.error('Audio processing failed:', error);
      setError('Failed to process audio. Please try again.');
    } finally {
      setIsProcessingAudio(false);
      options.onProcessingStateChange('idle');
    }
  }, [options, transcriptionMethod]);

  const analyzeAudioQuality = async (audioBlob: Blob) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    const response = await fetch('/api/voice/quality-analysis', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Audio quality analysis failed');
    }

    return await response.json();
  };

  const transcribeAudio = async (audioBlob: Blob, method: string, language: string) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    formData.append('language', language);
    formData.append('method', method);

    const response = await fetch('/api/voice/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Transcription failed: ${response.statusText}`);
    }

    return await response.json();
  };

  // Audio analysis utility functions
  const calculateRMSVolume = (dataArray: Uint8Array): number => {
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const sample = (dataArray[i] - 128) / 128;
      sum += sample * sample;
    }
    return Math.sqrt(sum / dataArray.length);
  };

  const estimatePitch = (dataArray: Uint8Array): number => {
    // Simplified pitch estimation using autocorrelation
    const autocorr = new Array(dataArray.length).fill(0);
    
    for (let lag = 0; lag < dataArray.length; lag++) {
      for (let i = 0; i < dataArray.length - lag; i++) {
        autocorr[lag] += dataArray[i] * dataArray[i + lag];
      }
    }
    
    // Find first peak after lag 20
    let maxVal = 0;
    let bestLag = 0;
    
    for (let lag = 20; lag < autocorr.length / 2; lag++) {
      if (autocorr[lag] > maxVal) {
        maxVal = autocorr[lag];
        bestLag = lag;
      }
    }
    
    return bestLag > 0 ? 44100 / bestLag : 0;
  };

  const analyzeSpeechLikeness = (frequencyData: Uint8Array): number => {
    // Analyze frequency distribution for speech characteristics
    const speechFreqRange = frequencyData.slice(10, 80);
    const avgAmplitude = speechFreqRange.reduce((a, b) => a + b, 0) / speechFreqRange.length;
    
    // Look for formant peaks
    const peaks = findSpectralPeaks(speechFreqRange);
    const speechScore = (avgAmplitude / 128) * Math.min(peaks.length / 3, 1);
    
    return Math.min(speechScore, 1.0);
  };

  const assessAudioQuality = (timeData: Uint8Array, freqData: Uint8Array): number => {
    // Signal-to-noise ratio
    const signal = calculateRMSVolume(timeData);
    const noise = Math.min(...Array.from(timeData).map(x => Math.abs(x - 128))) / 128;
    const snr = signal / Math.max(noise, 0.01);
    
    // Dynamic range
    const dynamicRange = (Math.max(...timeData) - Math.min(...timeData)) / 255;
    
    // Frequency distribution
    const freqSpread = freqData.reduce((acc, val, idx) => acc + val * idx, 0) / freqData.reduce((a, b) => a + b, 1);
    const normalizedSpread = Math.min(freqSpread / 1000, 1);
    
    return Math.min((snr * 0.4 + dynamicRange * 0.3 + normalizedSpread * 0.3), 1.0);
  };

  const detectVoiceEmotion = (timeData: Uint8Array, freqData: Uint8Array): Record<string, number> => {
    // Simplified emotion detection based on acoustic features
    const energy = calculateRMSVolume(timeData);
    const spectralCentroid = freqData.reduce((acc, val, idx) => acc + val * idx, 0) / freqData.reduce((a, b) => a + b, 1);
    const spectralVariance = freqData.reduce((acc, val, idx) => acc + Math.pow(idx - spectralCentroid, 2) * val, 0) / freqData.reduce((a, b) => a + b, 1);
    
    return {
      neutral: 0.4,
      concerned: Math.min(spectralVariance / 1000, 1.0),
      understanding: Math.min(energy * 2, 1.0),
      helpful: Math.min(spectralCentroid / 500, 1.0),
      empathetic: Math.min(energy * spectralVariance / 500, 1.0)
    };
  };

  const findSpectralPeaks = (data: Uint8Array): number[] => {
    const peaks: number[] = [];
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] && data[i] > data[i + 1] && data[i] > 50) {
        peaks.push(i);
      }
    }
    return peaks;
  };

  const determineOptimalTranscriptionMethod = (metrics: VoiceMetrics): 'whisper' | 'flamingo3' | 'hybrid' => {
    const qualityScore = metrics.quality * metrics.naturalness;
    
    if (qualityScore > 0.8) {
      return 'whisper'; // High quality → Whisper for best accuracy
    } else if (qualityScore > 0.5) {
      return 'hybrid'; // Medium quality → Try both methods
    } else {
      return 'flamingo3'; // Low quality → Flamingo 3 for noise handling
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
    };
  }, []);

  return {
    isListening,
    isProcessingAudio,
    audioLevel,
    voiceMetrics,
    error,
    transcriptionMethod,
    startListening,
    stopListening
  };
};

// Hook for WebSocket communication with natural conversation flow
export const useWebSocketConversation = (
  sessionId: string,
  language: string,
  wsEndpoint: string
) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState<'excellent' | 'good' | 'poor'>('excellent');
  const [conversationState, setConversationState] = useState({
    isUserTurn: true,
    expectingResponse: false,
    turnState: 'idle' as 'idle' | 'listening' | 'processing' | 'speaking'
  });
  const [messages, setMessages] = useState<any[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);

  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    try {
      const wsUrl = `${wsEndpoint}/ws/voice-c

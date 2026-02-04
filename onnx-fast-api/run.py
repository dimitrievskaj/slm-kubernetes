import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_ID = os.getenv("MODEL_ID", "EleutherAI/gpt-neo-125M")
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.onnx")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "30"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# Global model state
tokenizer = None
session = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global tokenizer, session, model_loaded
    logger.info(f"Loading tokenizer: {MODEL_ID}")
    logger.info(f"Loading ONNX model: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        session = ort.InferenceSession(MODEL_PATH)
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="LLM Text Generation API (ONNX)",
    description="FastAPI service for text generation using ONNX Runtime",
    version="1.0.0",
    lifespan=lifespan,
)


class PromptInput(BaseModel):
    prompt: str = Field(..., description="The input text to continue")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=512, description="Maximum tokens to generate")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="Top-k sampling parameter")


class GenerationResponse(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness probe - checks if the service is running."""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/ready", response_model=HealthResponse)
def readiness_check():
    """Readiness probe - checks if the service is ready to accept traffic."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_loaded": model_loaded}


@app.post("/generate", response_model=GenerationResponse)
def generate_text(data: PromptInput):
    """Generate text based on the input prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prompt = data.prompt
        max_tokens = data.max_tokens
        k = data.top_k

        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        generated_ids = input_ids.copy()

        for _ in range(max_tokens):
            seq_len = generated_ids.shape[1]
            position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

            outputs = session.run(None, {
                "input_ids": generated_ids,
                "attention_mask": np.ones_like(generated_ids),
                "position_ids": position_ids
            })

            logits = outputs[0][0, -1]

            top_k_ids = logits.argsort()[-k:][::-1]
            top_k_logits = logits[top_k_ids]
            probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
            next_token_id = int(np.random.choice(top_k_ids, p=probs))

            next_token = np.array([[next_token_id]])
            generated_ids = np.concatenate([generated_ids, next_token], axis=1)

        output_text = tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        ).strip()
        return {"text": output_text}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

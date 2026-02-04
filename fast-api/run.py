import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_ID = os.getenv("MODEL_ID", "EleutherAI/gpt-neo-125M")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "50"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "50"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.95"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.2"))

# Global model state
tokenizer = None
model = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global tokenizer, model, model_loaded
    logger.info(f"Loading model: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="LLM Text Generation API",
    description="FastAPI service for text generation using Hugging Face Transformers",
    version="1.0.0",
    lifespan=lifespan,
)


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The input text to continue")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.1, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="Top-k sampling parameter")
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")


class GenerationResponse(BaseModel):
    generated_text: str


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
def generate_text(request: PromptRequest):
    """Generate text based on the input prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

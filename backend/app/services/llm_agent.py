import json
from loguru import logger
from openai import OpenAI
from app.config import settings

# Fallback sequence of free models as requested
MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free", # user provided ID for Gemma 4 31B
    "qwen/qwen3-coder-480b-a35b-instruct:free", # user provided ID
    "openai/gpt-oss-20b:free"
]

def generate_chat_response(context_data: dict, user_message: str, chat_history: list = None) -> str:
    """
    Generates a response from the LLM assistant based on the diagnosis context and user query.
    """
    if not settings.openrouter_api_key:
        return "LLM API Key is missing. Please add OPENROUTER_API_KEY to your .env file."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )

    system_prompt = f"""
    You are an expert AI ophthalmologist embedded within the RETINA-Q clinical system. 
    You are helping a doctor interpret Retinal scans (OCT and Color Fundus) for a patient.
    Always be extremely concise, clinical, helpful, and easily understandable.
    
    Here is the exact diagnostic context from the current scan:
    - Image Modality: {context_data.get('image_type', 'Unknown')}
    - AI Prediction: {context_data.get('prediction', 'Unknown')}
    - Confidence: {context_data.get('confidence', 0.0) * 100:.2f}%
    
    If the prediction is CSR / CSCR, there is subretinal fluid or pigmentary distortion.
    If the prediction is Normal / Healthy, the tissue is structurally flat and normal.
    Answer the user's question focusing ONLY on this clinical scan context.
    """

    messages = [{"role": "system", "content": system_prompt.strip()}]
    
    if chat_history:
        for msg in chat_history:
            # chat_history format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    # Try models with fallback logic
    for model_id in MODELS:
        try:
            logger.info(f"Attempting LLM inference with OpenRouter model: {model_id}")
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Model {model_id} failed or busy: {e}")
            continue

    return "All free tier models are currently busy on OpenRouter. Please try answering your question again in a few moments."

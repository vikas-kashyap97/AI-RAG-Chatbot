import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ai_agent import get_response_from_ai_agent

# ✅ Load environment variables from .env (works locally only, not on Render)
load_dotenv()

# ✅ Create FastAPI app
app = FastAPI(title="LangGraph AI Agent")

# ✅ Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-multi-agent-chatbot.streamlit.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Supported models
ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192", 
    "llama-3.3-70b-versatile", 
    "gemini-1.5-flash-latest"
]

# ✅ Request Schema
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# ✅ Endpoint to handle incoming chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}

    try:
        print(f"📨 Received Request: {request.dict()}")

        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=request.messages,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Agent failed to respond: {str(e)}"}

# ✅ Run locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT in env
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Flask API server for local LLM
Wraps the LLMEngine to provide HTTP endpoints
"""
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.llm.engine import LLMEngine
from src.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global engine instance
engine = None
config = None

def initialize_engine():
    """Load the LLM model"""
    global engine, config
    log.info("🤖 Initializing LLM Engine...")
    config = Config()
    engine = LLMEngine(config)
    engine.load()
    log.info("✅ LLM Engine ready!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": config.model_id if config else "not loaded",
        "ready": engine is not None
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint
    Request format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', config.temperature)
        max_tokens = data.get('max_tokens', config.max_new_tokens)
        
        # Extract system prompt and user message
        system_prompt = config.chat_system_prompt
        history = []
        user_msg = ""
        
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'system':
                system_prompt = content
            elif role == 'user':
                user_msg = content
            elif role == 'assistant':
                # Add to history if there was a previous user message
                if user_msg:
                    history.append((user_msg, content))
                    user_msg = ""
        
        if not user_msg:
            return jsonify({"error": "No user message provided"}), 400
        
        # Generate response (non-streaming for simplicity)
        log.info(f"💬 Generating response for: {user_msg[:50]}...")
        response_text = ""
        for chunk in engine.generate_stream(
            system_prompt=system_prompt,
            history=history,
            user_msg=user_msg,
            max_new_tokens=max_tokens,
            temperature=temperature
        ):
            response_text += chunk
        
        log.info(f"✅ Response generated ({len(response_text)} chars)")
        
        # Return OpenAI-compatible format
        return jsonify({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "model": config.model_id,
            "usage": {
                "prompt_tokens": 0,  # Not calculated
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        log.error(f"❌ Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def main():
    """Run the Flask API server"""
    print("\n" + "="*60)
    print("🚀 Starting Local LLM API Server")
    print("="*60)
    
    initialize_engine()
    
    print("\n" + "="*60)
    print("✅ Server Running!")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"🤖 Model: {config.model_id}")
    print(f"🔍 Health check: http://localhost:5000/health")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
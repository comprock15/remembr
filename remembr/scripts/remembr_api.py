from flask import Flask, request, jsonify
import subprocess
import threading
import os
import logging
import base64
from datetime import datetime
from agents.remembr_agent import ReMEmbRAgent
from memory.milvus_memory import MilvusMemory

PARSE_SCRIPT_PATH = 'scripts/parse_video.py'
PREPROCESS_SCRIPT_PATH = 'scripts/preprocess_video.py'
PREPROCESS_CAPTIONS_SCRIPT_PATH = 'scripts/preprocess_captions.py'
UPLOAD_CAPTIONS_SCRIPT_PATH = 'scripts/upload_captions_to_memory.py'

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessRunner:
    def __init__(self, base_path="/home/docker_user/remembr/remembr"):
        self.current_process = None
        self.is_running = False
        self.base_path = base_path
    
    def run_scripts(self, video_path=None, collection_name="default_memory"):
        """Launch video processing scripts"""
        if self.is_running:
            return {"status": "error", "message": "Process is running already"}
        
        def run_in_thread():
            try:
                self.is_running = True
                logger.info(f"Launching video processing for collection: {collection_name}...") 
                             
                # 1. Парсинг видео
                if video_path and os.path.exists(video_path):
                    logger.info(f"Launching {PARSE_SCRIPT_PATH} for {video_path}")
                    result1 = subprocess.run([
                        "/opt/conda/envs/remembr/bin/python", PARSE_SCRIPT_PATH, 
                        "--input", video_path,
                        "--output", "tmp/data/parsed_data"
                    ], capture_output=True, text=True, cwd=self.base_path)
                    
                    if result1.returncode != 0:
                        logger.error(f"Error {PARSE_SCRIPT_PATH}: {result1.stderr}")
                        raise RuntimeError(f"Error {PARSE_SCRIPT_PATH}", result1.returncode)
                else:
                    logger.error(f"Error: video {video_path} not found")
                    raise FileNotFoundError(f"Error: video {video_path} not found")

                # 2. Препроцессинг
                logger.info(f"Launching {PREPROCESS_SCRIPT_PATH}")
                result2 = subprocess.run([
                    "/opt/conda/envs/remembr/bin/python", PREPROCESS_SCRIPT_PATH,
                    "--input", "tmp/data/parsed_data",
                    "--output", "tmp/data/preprocessed_data"
                ], capture_output=True, text=True, cwd=self.base_path)
                
                if result2.returncode != 0:
                    logger.error(f"Error {PREPROCESS_SCRIPT_PATH}: {result2.stderr}")
                    raise RuntimeError(f"Error {PARSE_SCRIPT_PATH}", result2.returncode)
               
                
                # 3. Обработка captions
                logger.info(f"Launching {PREPROCESS_CAPTIONS_SCRIPT_PATH}")
                result3 = subprocess.run([
                    "/opt/conda/envs/remembr/bin/python", PREPROCESS_CAPTIONS_SCRIPT_PATH,
                    "--seconds_per_caption", "5",
                    "--model-path", "Efficient-Large-Model/VILA1.5-13b",
                    "--captioner_name", "VILA1.5-13b",
                    "--conv-mode", "vicuna_v1",
                    "--data_path", "tmp/data/preprocessed_data", 
                    "--out_path", "tmp/data/captions"
                ], capture_output=True, text=True, cwd=self.base_path,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}
                )

                if result3.returncode != 0:
                    logger.error(f"Error {PREPROCESS_CAPTIONS_SCRIPT_PATH}: {result3.stderr}")
                    raise RuntimeError(f"Error {PREPROCESS_CAPTIONS_SCRIPT_PATH}", result3.returncode)

                # 4. Загрузка captions в память
                logger.info(f"Launching {UPLOAD_CAPTIONS_SCRIPT_PATH}")
                result4 = subprocess.run([
                    "/opt/conda/envs/remembr/bin/python", UPLOAD_CAPTIONS_SCRIPT_PATH,
                    "--captions", "tmp/data/captions/captions_VILA1.5-13b_5_secs.json",
                    "--collection", collection_name
                ], capture_output=True, text=True, cwd=self.base_path)

                if result4.returncode != 0:
                    logger.error(f"Error {UPLOAD_CAPTIONS_SCRIPT_PATH}: {result4.stderr}")
                    raise RuntimeError(f"Error {UPLOAD_CAPTIONS_SCRIPT_PATH}", result4.returncode)
                
                self.is_running = False
                logger.info(f"Processing ended successfully, data is stored to collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Error while processing: {e}")
                self.is_running = False
        
        thread = threading.Thread(target=run_in_thread)
        thread.daemon = True
        thread.start()
        
        return {"status": "success", "message": "Processing is in progress"}

class ReMEmbRAgentManager:
    def __init__(self, db_ip="127.0.0.1", llm_type="llama3.1:8b"):
        self.db_ip = db_ip
        self.llm_type = llm_type
        self.agents = {}  # Храним агентов для разных коллекций
        logger.info(f"Initializing ReMEmbR Agent Manager with DB: {db_ip}, LLM: {llm_type}")
    
    def get_agent(self, collection_name):
        """Get иor create agent for collection"""
        if collection_name not in self.agents:
            logger.info(f"Creating agent for collection: {collection_name}")
            try:
                memory = MilvusMemory(collection_name, db_ip=self.db_ip)
                agent = ReMEmbRAgent(llm_type=self.llm_type)
                agent.set_memory(memory)
                self.agents[collection_name] = agent
                logger.info(f"Agent for collection {collection_name} is created")
            except Exception as e:
                logger.error(f"Error while creating agent for {collection_name}: {e}")
                raise
        return self.agents[collection_name]
    
    def query_agent(self, collection_name, question):
        """Send request to agent"""
        try:
            agent = self.get_agent(collection_name)
            response = agent.query(question)
            
            result = {
                "status": "success",
                "question": question,
                "collection": collection_name
            }
            
            if hasattr(response, 'text') and response.text:
                result["answer"] = response.text
            
            if hasattr(response, 'position') and response.position:
                result["position"] = {
                    "x": float(response.position[0]),
                    "y": float(response.position[1]), 
                    "z": float(response.position[2])
                }
            
            logger.info(f"Successful request to collection {collection_name}: '{question}'")
            return result
            
        except Exception as e:
            logger.error(f"Error while sending request to {collection_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question,
                "collection": collection_name
            }

runner = ProcessRunner()
agent_manager = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# === Endpoints for video processing ===

@app.route('/process/video', methods=['POST'])
def process_video():
    data = request.get_json()

    if data.get('video_data'):
        video_data = base64.b64decode(data['video_data'])
        filename = data.get('filename', 'video.mp4')
        video_path = f"/tmp/{filename}"

        with open(video_path, "wb") as f:
            f.write(video_data)
    else:
        video_path = None
    
    result = runner.run_scripts(video_path)
    return jsonify(result)

@app.route('/process/status', methods=['GET'])
def get_status():
    return jsonify({
        "is_running": runner.is_running,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/process/cancel', methods=['POST'])
def cancel_process():
    if runner.current_process:
        runner.current_process.terminate()
    runner.is_running = False
    return jsonify({"status": "success", "message": "Process is cancelled"})

# === Endpoints for agent ===

@app.route('/agent/query', methods=['POST'])
def query_agent():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No JSON data provided"}), 400
        
        collection_name = data.get('collection', 'default_memory')
        question = data.get('question', '')
        
        if not question:
            return jsonify({"status": "error", "error": "No question provided"}), 400
        
        result = agent_manager.query_agent(collection_name, question)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in agent's API: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ReMEmbR Unified API Server")
    parser.add_argument("--db_ip", type=str, default="127.0.0.1", help="Milvus DB IP address")
    parser.add_argument("--llm_type", type=str, default="llama3.1:8b", help="LLM backend type")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind")
    
    args = parser.parse_args()
    
    # Инициализация менеджера агентов
    global agent_manager
    agent_manager = ReMEmbRAgentManager(db_ip=args.db_ip, llm_type=args.llm_type)
    
    logger.info(f"Launching ReMEmbR Unified API на {args.host}:{args.port}")
    logger.info(f"DB: {args.db_ip}, LLM: {args.llm_type}")
    logger.info("Available endpoints:")
    logger.info("  POST /process/video    - Video processing")
    logger.info("  GET  /process/status   - Process status")
    logger.info("  POST /process/cancel   - Cancel current processing")
    logger.info("  POST /agent/query      - Send request to agent")
    
    app.run(host=args.host, port=args.port, debug=True)

if __name__ == '__main__':
    main()
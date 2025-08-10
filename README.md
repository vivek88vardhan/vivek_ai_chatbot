# vivek_ai_chatbot
This is boilerplate template for chatbot with local llm and RAG, Vector DB


chatbot-project/
├── frontend/
├── middleware/
├── llm_engine/
├── local_api/
├── rag_engine/
├── vector_db/
├── prefect_flows/


frontend/app.py	                ✅ Streamlit UI
middleware/main.py	            ✅ FastAPI logic
middleware/session_manager.py	✅ Session manager
llm_engine/ollama_client.py	    ✅ Ollama client
local_api/api.py	            ✅ Flask API
rag_engine/retriever.py	        ✅ RAG retriever
vector_db/docker-compose.yml	✅ Weaviate setup
prefect_flows/chatbot_flow.py	✅ Prefect DAG

python -m venv myenv
in case of windows .\myenv\Scripts\activate.bat
in case of Mac or Linux myenv/bin/activate

pip install prefect weaviate-client sentence-transformers numpy


Prerequisites:
1. Docker with Weaviate running: docker run -d --name weaviate -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:latest
2. Install dependencies: pip install prefect weaviate-client sentence-transformers numpy
3. Start Prefect server: prefect server start
4. In another terminal, run this script: python weaviate_test_prefect.py

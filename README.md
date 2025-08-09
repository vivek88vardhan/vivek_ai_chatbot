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


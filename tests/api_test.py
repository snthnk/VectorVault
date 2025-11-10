import uvicorn
from src.main import app
import requests
from tests.test_persistance import texts_list as texts_list1
from tests.vector_search_test import texts_list as texts_list2

texts_list = texts_list1 + texts_list2

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
    requests.get("http://127.0.0.1:8000/health")
    requests.get("http://127.0.0.1:8000/clear")
    payload = {
        "documents":[
            {
                "text": text,
                "doc_id": None,
                "metadata": {}
            }
        ] for text in texts_list
    }
    requests.get("http://127.0.0.1:8000/index", json=payload)
    requests.get("http://127.0.0.1:8000/health")
    query = "искусственный интеллект"
    payload = {
        "query": query
    }
    requests.get("http://127.0.0.1:8000/search", json=payload)

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn
import requests
import tempfile
from pipeline import initialize_pipeline, answer_question_pipeline

app = FastAPI()

class QARequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QAResponse)
def run_pipeline(request: QARequest):
    try:
        response = requests.get(request.documents)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download document")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            file_path = tmp_file.name

        hybrid_retriever, bm25_retriever, llm, reranker = initialize_pipeline(file_path)
        
        answers = []
        for question in request.questions:
            answer = answer_question_pipeline(
                question, hybrid_retriever, bm25_retriever, llm, reranker
            )
            answers.append(answer)

        return {"answers": answers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

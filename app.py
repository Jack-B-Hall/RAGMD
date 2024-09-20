from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from query_data import query_rag

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>RAG AI Model Query Interface</title>
        </head>
        <body>
            <h1>RAG AI Query Interface</h1>
            <form action="/query/" method="post">
                <input type="text" name="query_text" size="50">
                <button type="submit">Submit Query</button>
            </form>
        </body>
    </html>
    """
    return html_content

@app.post("/query/")
async def query_model(query_text: str = Form(...)):
    try:
        response = query_rag(query_text)
        return {"Query": query_text, "Response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

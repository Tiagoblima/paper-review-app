from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from app.agents.review_agent import PaperReviewAgent
from app.core.services.review_service import ReviewService
DEFAUTL_SAVE_PATH = "app/resources/uploads"
app = FastAPI()
research_questions = ["What are the techniques used to apply bloom taxonomy?",
                      "What are objectives of applying bloom taxonomy?",
                      "What Bloom taxonomy level is investigated?",
                      "What are the target disciplines and audiance?",
                      " What is the datasets used?",
                      "What are the qualitative and quantitative impirical evidences?",
                      " What are the current challenges when applying LLMs to build Bloom Taxonomy aligned questions?"]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_files(files: Annotated[list[bytes], File()]):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):

    results = []
    for file in files:
        contents = await file.read()
        with open(f"{DEFAUTL_SAVE_PATH}/{file.filename}", "wb") as f:
            f.write(contents)

        review_service = ReviewService(f"{DEFAUTL_SAVE_PATH}/{file.filename}")
        for i, question in enumerate(research_questions):
            result = review_service.invoke(question)
            result["context"] = [doc.page_content for doc in result["context"]]
            result["source"] = file.filename
            results.append(result)
    return results


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
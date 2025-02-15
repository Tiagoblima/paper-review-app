from pprint import pprint
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from app.agents.review_agent import PaperReviewAgent
from app.core.services.chat_service import ChatModelService
from app.core.services.processing_service import ProcessingService
from app.core.services.vectorstore_service import VectorStoreService
DEFAUTL_SAVE_PATH = "./resources/uploads"
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_files(files: Annotated[list[bytes], File()]):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):

    for file in files:
        contents = await file.read()
        with open(f"{DEFAUTL_SAVE_PATH}/{file.filename}", "wb") as f:
            f.write(contents)

        processing_service = ProcessingService()
        pages = processing_service.process_paper(f"{DEFAUTL_SAVE_PATH}/{file.filename}")
        vectorstore_service = VectorStoreService()
        vectorstore_service.add_documents(pages)
        chat_service = ChatModelService()
        review_agent = PaperReviewAgent(chat_service, vectorstore_service)
        review_agent.start()
        review_agent.build_agent()
        answer = review_agent.graph.invoke({"question": "What is the main idea of the paper?"})
        pprint(answer)
    return {"filenames": [file.filename for file in files]}


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
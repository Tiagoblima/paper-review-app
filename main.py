from dotenv import load_dotenv
import pandas as pd
from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from app.core.services.review_service import ReviewService
from app.utils.utils import save_df_to_excel
DEFAUTL_SAVE_PATH = "app/resources/uploads"
load_dotenv()
app = FastAPI()
research_questions = ["What are the techniques used to apply bloom taxonomy?",
                      "What are objectives of applying bloom taxonomy?",
                      "What Bloom taxonomy level is investigated?",
                      "What are the target disciplines and audiance?",
                      " What is the datasets used?",
                      "What are the qualitative and quantitative impirical evidences?",
                      "What are the current challenges when applying LLMs to build Bloom Taxonomy aligned questions?"]

basic_info_keys = ["title", "authors", "year", "abstract", "keywords", "doi", "country", "conference"]

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_files(files: Annotated[list[bytes], File()]):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):

    results = []
    basic_info_list = []
    for paper_id, file in enumerate(files, 1):
        contents = await file.read()
        with open(f"{DEFAUTL_SAVE_PATH}/{file.filename}", "wb") as f:
            f.write(contents)

        review_service = ReviewService(f"{DEFAUTL_SAVE_PATH}/{file.filename}", basic_info_keys)
        basic_info = review_service.get_basic_info()
        basic_info["source"] = file.filename
        
        filtered_basic_info = {key: basic_info[key] for key in basic_info_keys if key in basic_info}
        filtered_basic_info["ID"] = paper_id
        save_df_to_excel(pd.DataFrame.from_dict(filtered_basic_info, orient="index").T,
                          "Basic Info", f"{DEFAUTL_SAVE_PATH}/results.xlsx")
        basic_info_list.append(filtered_basic_info)
        for i, question in enumerate(research_questions):
            result = review_service.invoke(question)
            result["context"] = [doc.page_content for doc in result["context"]]
            result["source"] = file.filename
            results.append(result)
            #save_df_to_excel(result, f"RQ{i+1}", f"{DEFAUTL_SAVE_PATH}/results.xlsx")
        
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
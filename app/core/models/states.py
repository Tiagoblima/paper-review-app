from typing import List
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    supporting_text: str


class BasicInfoState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    conference: str
    title: str
    authors: str
    year: str
    abstract: str
    keywords: str
    doi: str
    country: str
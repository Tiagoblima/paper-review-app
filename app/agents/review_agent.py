

import json
from app.core.models.states import State
from app.core.repository.models import ChatModelRepository, EmbeddingsModelRepository
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from app.core.services.chat_service import ChatModelService
from app.core.services.vectorstore_service import VectorStoreService




class PaperReviewAgent:
    def __init__(self, chat_service: ChatModelService, 
                 vectorstore_service: VectorStoreService):
        self.chat_service = chat_service
        self.vectorstore_service = vectorstore_service
        self.json_prompts = json.load(open("resources/prompts.json"))

    def start(self):
        self._default_system_prompt = self.json_prompts["backbone_prompt"]
        self._few_shot_prompt = self.json_prompts["few_shot_prompt"]
        self._response_prompt = self.json_prompts["response_prompt"]

        self.prompt_template = PromptTemplate(
            template=self._default_system_prompt,
            input_variables=["context", "question"]
        )
    
    def retrieve(self,state: State):
        retrieved_docs = self.vectorstore_service.retrieve(state["question"])
        return {"context": retrieved_docs}


    def generate(self,state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt_template.invoke({"question": state["question"], "context": docs_content})
        response = self.chat_service.invoke(messages)
        return {"answer": response.content}
    
    def build_agent(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        return self.graph
    
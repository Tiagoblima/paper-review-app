

import json
from langchain_core.messages import BaseMessage
from bs4 import BeautifulSoup
from app.core.models.states import BasicInfoState, State
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from app.core.services.chat_service import ChatModelService
from app.core.services.vectorstore_service import VectorStoreService
from pydantic import BaseModel, Field
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = Field(description="The answer to the user's question")
    supporting_text: str = Field(description="A text from the context that supports the answer.")



class BasicInfoAgent:
    def __init__(self, chat_service: ChatModelService, 
                 vectorstore_service: VectorStoreService, info_keys: list[str]):
        self.chat_service = chat_service
        self.vectorstore_service = vectorstore_service
        self.json_prompts = json.load(open("app/resources/prompts.json"))
        self._default_system_prompt = self.json_prompts["backbone_prompt"]
        self._few_shot_prompt = self.json_prompts["few_shot_prompt"]
        self._response_prompt = self.json_prompts["response_prompt"]
        self.info_keys = info_keys
        self.prompt_template = PromptTemplate.from_template(
            """
                You are my research assistent and I need you to help me extract certain information according to some research question. 
                {context}
                Return the answer in a structured format only with the information requested.
                Also, you need to provide within quotes the piece of text in the paper that supports the answer.
                Question: {question}
                RETURN the basic information of the paper in the following format:
                <title> title </title>
                <authors> authors </authors>
                <year> year </year>
                <abstract> abstract </abstract>
                <keywords> keywords </keywords>
                <doi> doi </doi>
                <country> country </country>
                <conference> conference </conference>
            """)
    
    def parse_response(self, state: BasicInfoState):
        
        soup = BeautifulSoup(state["answer"], 'html.parser')
        info_dict = {}
        for info in self.info_keys:
            if soup.find(info):
                info_dict[info] = soup.find(info).get_text().strip("\n")
            else:
                info_dict[info] = "Not found"
        
        return info_dict
    
    def retrieve(self,state: BasicInfoState):
        print(f"Retrieving context for question: {state['question']}")
        retrieved_docs = self.vectorstore_service.retrieve(state["question"])
        return {"context": retrieved_docs}


    def generate(self,state: BasicInfoState):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt_template.invoke({"question": state["question"], "context": docs_content})
        response = self.chat_service.invoke(messages)
        
        
        return {"answer": response.content}
    
    def build_agent(self):
        graph_builder = StateGraph(BasicInfoState).add_sequence([self.retrieve, self.generate, self.parse_response])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        return self.graph
    
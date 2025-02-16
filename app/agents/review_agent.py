

import json
from langchain_core.messages import BaseMessage
from bs4 import BeautifulSoup
from app.core.models.states import State
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from app.core.services.chat_service import ChatModelService
from app.core.services.vectorstore_service import VectorStoreService
from pydantic import BaseModel, Field
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = Field(description="The answer to the user's question")
    supporting_text: str = Field(description="A text from the context that supports the answer.")



class PaperReviewAgent:
    def __init__(self, chat_service: ChatModelService, 
                 vectorstore_service: VectorStoreService):
        self.chat_service = chat_service
        self.vectorstore_service = vectorstore_service
        self.json_prompts = json.load(open("app/resources/prompts.json"))
        self._default_system_prompt = self.json_prompts["backbone_prompt"]
        self._few_shot_prompt = self.json_prompts["few_shot_prompt"]
        self._response_prompt = self.json_prompts["response_prompt"]

        self.prompt_template = PromptTemplate.from_template(
            """
                You are my research assistent and I need you to help me extract certain information according to some research question. 
                {context}
                Return the answer in a structured format only with the information requested.
                Also, you need to provide within quotes the piece of text in the paper that supports the answer.
                Question: {question}
                RETURN the answer and the supporting text in the following format:
                <answer> answer </answer>
                <supporting_text> supporting_text </supporting_text>


                Example:
                Question: What are the techniques used to apply bloom taxonomy?
                <answer> The techniques used to apply bloom taxonomy are: </answer>
                <supporting_text> Supporting text: </supporting_text>

            """)
    
    def parse_response(self, state: State):
        print(f"State: {state}")
        soup = BeautifulSoup(state["answer"], 'html.parser')
        #print(f"Soup: {soup.prettify()}"
        return {
            "answer": soup.find("answer").get_text(),
            "supporting_text": soup.find("supporting_text").get_text()
        }
    def retrieve(self,state: State):
        print(f"Retrieving context for question: {state['question']}")
        retrieved_docs = self.vectorstore_service.retrieve(state["question"])
        return {"context": retrieved_docs}


    def generate(self,state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt_template.invoke({"question": state["question"], "context": docs_content})
        response = self.chat_service.invoke(messages)
        #print(f"Response: {response.content}")
        
        return {"answer": response.content}
    
    def build_agent(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate, self.parse_response])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()
        return self.graph
    
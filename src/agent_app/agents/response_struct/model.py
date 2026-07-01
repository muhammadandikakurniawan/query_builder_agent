from typing import TypedDict, Optional, Type, Union

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

from agent_app.shared.utils.content import remove_think_blocks


class AgentState(TypedDict):
    llm: ChatOpenAI
    input_prompts: list[BaseMessage]
    schema_cls: Type[BaseModel]

    structured: Optional[Union[str, dict]]
    error: Optional[str]

    retries: int
    max_retry: int
    output: Optional[dict]
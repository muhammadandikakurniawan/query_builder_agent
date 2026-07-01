from typing import TypeVar, TypedDict, Optional, Type, Union

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
# from agent_app.shared.logging.logger import get_logger
from agent_app.shared.model.response import Result
from agent_app.agents.response_struct.model import AgentState
from agent_app.agents.response_struct.nodes import ResponseStructAgentNodes
from agent_app.shared.utils.content import remove_think_blocks




# =====================================================
# Agent Builder
# =====================================================
SCHEMA_CLS = TypeVar("T", bound=BaseModel)
class ResponseStructAgent:
    def __init__(self):
        nodes = ResponseStructAgentNodes()
        graph = StateGraph(AgentState)

        graph.add_node("extract",lambda state: nodes.extract_node(state=state))
        graph.add_node("validate",nodes.validate_node)
        graph.add_node( "repair",lambda state: nodes.repair_node(state=state))

        graph.set_entry_point("extract")
        graph.add_edge("extract", "validate")
        graph.add_conditional_edges("validate",nodes.route)
        graph.add_edge("repair","validate")

        self.agent = graph.compile()
        pass

    def invoke(self, prompts: list[tuple[str, str]], llm: ChatOpenAI, schema_cls: Type[SCHEMA_CLS], max_retry: int) -> Result[SCHEMA_CLS]:
        try:
            result = self.agent.invoke(AgentState(
                llm=llm,
                input_prompts=prompts,
                schema_cls=schema_cls,
                max_retry=max_retry,
                retries=0
            ))

            if result["error"]:
                return Result.fail(code="AGENT_ERROR", message=str(result["error"]))

            if result["output"] is None:
                return Result.fail(code="AGENT_ERROR", message="failed validate output")

            return Result.ok(schema_cls.model_validate(result["output"]))

        except Exception as ex:
            # get_logger(__name__).exception(str(ex))
            return Result.fail(code="AGENT_ERROR", message=str(ex))


from typing import Optional, Type, Union

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, ValidationError

from agent_app.agents.response_struct.model import AgentState
from agent_app.shared.utils.content import extract_json, remove_think_blocks


# =====================================================
# State
# =====================================================



class ResponseStructAgentNodes:
    def __init__(self):
        pass

    # =====================================================
    # Nodes
    # =====================================================

    def extract_node(self, state: AgentState):
        schema_cls = state["schema_cls"]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
        You are a data extraction engine.

        Return only valid JSON.
        Schema:
        {schema}
        """
            ),
            MessagesPlaceholder("messages"),
        ])

        msg = prompt.format_messages(
            schema=schema_cls.model_json_schema(),
            messages=state["input_prompts"],
        )

        result = state["llm"].invoke(msg)
        result.content = extract_json(remove_think_blocks(result.content))
        return {
            "structured": result.content,
            "error": None,
        }


    def validate_node(self, state: AgentState):
        schema_cls = state["schema_cls"]

        try:
            obj = schema_cls.model_validate_json(
                state["structured"]
            )

            return {
                "output": obj, 
                "structured": obj.model_dump(),
                "error": None,
            }

        except ValidationError as e:
            return {
                "error": str(e)
            }


    def repair_node(self, state: AgentState):
        schema_cls = state["schema_cls"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Fix this output so it exactly matches the schema.

            JSON Schema:
            {schema}

            Bad Output:
            {output}

            Return ONLY valid JSON.
            """),
            MessagesPlaceholder("messages")
        ])

        msg = prompt.format_messages(
            schema=schema_cls.model_json_schema(),
            output=state["structured"],
            messages= state["input_prompts"]
        )

        result = state["llm"].invoke(msg)
        result.content = extract_json(remove_think_blocks(result.content))
        return {
            "structured": result.content,
            "retries": state["retries"] + 1,
        }


    def route(self, state: AgentState):
        if state["error"]:
            if state["retries"] >= state["max_retry"]:
                return END

            return "repair"

        return END
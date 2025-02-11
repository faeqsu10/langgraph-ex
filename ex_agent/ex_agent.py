from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 도구 정의
@tool
def multiply(a: int, b: int) -> int:
    """a와 b를 곱합니다.

    인자:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """a와 b를 더합니다.

    인자:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """a를 b로 나눕니다.

    인자:
        a: 첫 번째 정수
        b: 두 번째 정수
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# 노드
def llm_call(state: MessagesState):
    """LLM이 도구를 호출할지 여부를 결정합니다"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="당신은 일련의 입력값에 대해 산술 연산을 수행하는 도움이 되는 어시스턴트입니다."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """도구 호출을 수행합니다"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# LLM이 도구 호출을 했는지 여부에 따라 도구 노드로 이동할지 종료할지 결정하는 조건부 엣지 함수
def should_continue(state: MessagesState) -> Literal["environment", END]:
    """LLM이 도구 호출을 했는지 여부에 따라 루프를 계속할지 중단할지 결정합니다"""

    messages = state["messages"]
    last_message = messages[-1]
    # LLM이 도구를 호출하면 액션을 수행
    if last_message.tool_calls:
        return "Action"
    # 그렇지 않으면 중단 (사용자에게 응답)
    return END


# 워크플로우 구축
agent_builder = StateGraph(MessagesState)

# 노드 추가
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# 노드를 연결하는 엣지 추가
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # should_continue가 반환한 이름 : 다음에 방문할 노드의 이름
        "Action": "environment",
        END: END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# 에이전트 컴파일
agent = agent_builder.compile()

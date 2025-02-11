from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# 라우팅 로직을 위한 구조화된 출력 스키마
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="다음 단계의 라우팅 프로세스"
    )


# LLM에 구조화된 출력 스키마 적용
router = llm.with_structured_output(Route)


# 상태 정의
class State(TypedDict):
    input: str  # 사용자 입력
    decision: str  # 라우팅 결정
    output: str  # 최종 출력


# 노드 정의
def llm_call_1(state: State):
    """이야기 작성"""

    result = llm.invoke(f"{state['input']}에 대한 짧은 이야기를 써주세요.")
    return {"output": result.content}


def llm_call_2(state: State):
    """농담 작성"""

    result = llm.invoke(f"{state['input']}에 대한 재미있는 농담을 해주세요.")
    return {"output": result.content}


def llm_call_3(state: State):
    """시 작성"""

    result = llm.invoke(f"{state['input']}에 대한 시를 써주세요.")
    return {"output": result.content}


def llm_call_router(state: State):
    """입력을 적절한 노드로 라우팅"""

    # 라우팅 로직을 위한 구조화된 출력 LLM 실행
    decision = router.invoke(
        [
            SystemMessage(
                content="사용자의 요청에 따라 이야기(story), 농담(joke), 시(poem) 중 하나로 라우팅하세요."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


# 조건부 라우팅을 위한 결정 함수
def route_decision(state: State):
    # 다음에 방문할 노드 이름 반환
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# 워크플로우 구축
router_builder = StateGraph(State)

# 노드 추가
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# 노드 연결을 위한 엣지 추가
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # route_decision이 반환한 이름 : 다음에 방문할 노드 이름
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

graph = router_builder.compile()

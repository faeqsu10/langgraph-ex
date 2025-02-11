from typing import TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from langgraph.graph import StateGraph, START, END
llm = ChatOpenAI(model="gpt-4", temperature=0)


# 그래프 상태
class State(TypedDict):
    joke: str  # 농담
    topic: str  # 주제
    feedback: str  # 피드백
    funny_or_not: str  # 재미있는지 여부


# 평가를 위한 구조화된 출력 스키마
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="농담이 재미있는지 아닌지 판단하세요.",
    )
    feedback: str = Field(
        description="농담이 재미없다면, 개선을 위한 피드백을 제공하세요.",
    )


# LLM에 구조화된 출력 스키마 적용
evaluator = llm.with_structured_output(Feedback)


# 노드 정의
def llm_call_generator(state: State):
    """농담 생성기"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"{state['topic']}에 대한 농담을 작성하되, 다음 피드백을 반영해주세요: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"{state['topic']}에 대한 재미있는 농담을 만들어주세요.")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """농담 평가기"""

    grade = evaluator.invoke(f"다음 농담을 평가해주세요: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# 평가 결과에 따라 농담 생성기로 돌아가거나 종료하는 조건부 라우팅 함수
def route_joke(state: State):
    """평가 결과에 따른 라우팅"""

    # TODO: 랜덤하게 승인 또는 거부
    # if random.random() < 0.3:
    #     return "Accepted"
    # else:
    #     return "Rejected + Feedback"

    if state["funny_or_not"] == "funny":
        return "Accepted"  # 승인됨
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"  # 거부됨 + 피드백


# 워크플로우 구축
optimizer_builder = StateGraph(State)

# 노드 추가
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# 노드 연결을 위한 엣지 추가
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # route_joke가 반환한 값 : 다음에 방문할 노드
        "Accepted": END,  # 승인된 경우 종료
        "Rejected + Feedback": "llm_call_generator",  # 거부된 경우 다시 생성
    },
)

# 워크플로우 컴파일
graph = optimizer_builder.compile()

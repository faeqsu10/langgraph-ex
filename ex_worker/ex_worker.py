from langgraph.constants import Send
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
import operator
from typing import Annotated, List
from pydantic import BaseModel, Field
# 테스트: 인공지능의 현재와 미래

llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# 보고서 섹션 정의를 위한 스키마
class Section(BaseModel):
    name: str = Field(
        description="보고서 섹션의 제목",
    )
    description: str = Field(
        description="이 섹션에서 다룰 주요 주제와 개념에 대한 간단한 개요",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="보고서의 섹션들",
    )


# LLM에 구조화된 출력 스키마 적용
planner = llm.with_structured_output(Sections)


# 그래프 상태
class State(TypedDict):
    topic: str  # 보고서 주제
    sections: list[Section]  # 보고서 섹션 리스트
    completed_sections: Annotated[
        list, operator.add
    ]  # 모든 작업자가 병렬로 작성하는 섹션
    final_report: str  # 최종 보고서


# 작업자 상태
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# 노드 정의
def orchestrator(state: State):
    """보고서 계획을 생성하는 오케스트레이터"""

    # 섹션 생성
    report_sections = planner.invoke(
        [
            SystemMessage(content="주어진 주제에 대한 보고서 계획을 생성하세요."),
            HumanMessage(content=f"보고서 주제: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """보고서 섹션을 작성하는 작업자"""

    # 섹션 생성
    section = llm.invoke(
        [
            SystemMessage(
                content="주어진 제목과 설명에 따라 보고서 섹션을 작성하세요. 서문 없이 바로 내용을 시작하고, 마크다운 형식을 사용하세요."
            ),
            HumanMessage(
                content=f"섹션 제목: {state['section'].name}\n설명: {state['section'].description}"
            ),
        ]
    )

    # 완성된 섹션 반환
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """섹션들을 하나의 완성된 보고서로 통합"""

    # 완성된 섹션 목록
    completed_sections = state["completed_sections"]

    # 섹션들을 구분선으로 연결하여 최종 보고서 생성
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# 각 섹션별 작업자 할당을 위한 조건부 엣지 함수
def assign_workers(state: State):
    """각 섹션에 작업자 할당"""

    # Send() API를 통해 섹션 작성을 병렬로 시작
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# 워크플로우 구축
orchestrator_worker_builder = StateGraph(State)

# 노드 추가
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# 노드 연결을 위한 엣지 추가
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# 워크플로우 컴파일
graph = orchestrator_worker_builder.compile()

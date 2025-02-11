from typing import TypedDict
from langgraph.graph import StateGraph, END, START

"""
LangGraph의 가장 기본적인 예제

기능:
- 숫자를 1씩 증가시키는 단순한 카운터
- START → ADD_ONE → END 순서로 실행

구현 내용:
1. TypedDict로 상태 정의 (count 필드)
2. StateGraph 인스턴스 생성
3. 상태를 변경하는 노드 함수 구현 (add_one)
4. 노드 추가 및 엣지 연결

실행 결과:
- 입력: { count: n }
- 출력: { count: n + 1 }
"""


# 1. 상태 정의
class CounterState(TypedDict, total=False):
    messages: list[str]
    new_message: str


# 2. StateGraph 인스턴스 생성
# CounterState를 사용하여 그래프 초기화
builder = StateGraph(CounterState)


# 3. 노드 함수 구현
# 상태의 count 값을 1 증가시키는 함수
async def add_one(state: CounterState) -> CounterState:
    messages = state.get("messages", [])
    new_message = state["new_message"]
    return {"messages": messages + [new_message], "new_message": new_message}


# 4. 그래프 구성
# START → ADD_ONE → END 순서로 노드 연결
builder.add_node("ADD_ONE", add_one)
builder.add_edge(START, "ADD_ONE")
builder.add_edge("ADD_ONE", END)

# 5. 그래프 컴파일
graph = builder.compile() 
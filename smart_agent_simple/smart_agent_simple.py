from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from utils.utils import DebugChatPromptTemplate

#
# HUMAN: 안녕? 넌 누구니?
# HUMAN: 복리이자 계산해줘
# HUMAN: 원금 5만원, 이율 3%
# HUMAN: 10년
# (TavilySearch 검색 수 해당 State로 부터 실행)
# HUMAN: 괜찮은 상품을 추천해 줄래?
# HUMAN: 수익률이 가장 높은 상품의 예상 수익을 알려줘
#

# 1. 상태 정의
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 2. LLM 설정
llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# 3. 도구 정의
@tool
def compound_interest_calculator(
    principal: float, annual_rate: float, years: float, compounds_per_year: int = 12
) -> dict:
    """
    복리 이자를 계산합니다.

    Args:
        principal (float): 원금
        annual_rate (float): 연이율 (예: 5.0은 5%를 의미)
        years (float): 투자 기간 (년)
        compounds_per_year (int): 연간 복리 계산 횟수 (기본값 12, 월복리)

    Returns:
        dict: 계산 결과 정보
            - principal: 원금
            - total_amount: 만기 금액
            - interest_earned: 이자 수익
            - effective_rate: 실효 수익률
    """
    rate_decimal = annual_rate / 100
    total_amount = principal * (1 + rate_decimal / compounds_per_year) ** (
        compounds_per_year * years
    )
    interest_earned = total_amount - principal
    effective_rate = ((total_amount / principal) ** (1 / years) - 1) * 100

    return {
        "principal": round(principal, 2),
        "total_amount": round(total_amount, 2),
        "interest_earned": round(interest_earned, 2),
        "effective_rate": round(effective_rate, 2),
    }


# 4. 프롬프트 설정
assistant_prompt = DebugChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 금융 상담 전문가입니다.
            고객으로부터 다음 정보를 얻어야 합니다:
            - 투자하실 원금
            - 적용되는 연이율
            - 투자 기간 (년)
            - 복리 계산 주기 (선택사항, 기본값은 월복리)
            
            이 정보를 명확하게 파악할 수 없다면 고객에게 물어보세요! 추측하지 마세요.
            
            모든 정보를 파악했다면 복리 이자 계산기를 사용하여 다음 정보를 안내해주세요:
            - 원금
            - 만기 시 받으실 금액
            - 이자 수익
            - 실효 수익률
            """,
        ),
        ("placeholder", "{messages}"),
    ]
)


# 5. 도구 및 어시스턴트(Agent) 설정
available_tools = [TavilySearchResults(max_results=5), compound_interest_calculator]
assistant = assistant_prompt | llm.bind_tools(available_tools)


# 6. 어시스턴트 함수 정의
def process_message(state: State):
    """
    현재 대화 상태를 처리하고 적절한 응답을 생성합니다.

    Args:
        state (State): 현재 대화 상태

    Returns:
        dict: 업데이트된 메시지를 포함한 새로운 상태
    """
    result = assistant.invoke(state)

    # 유효하지 않은 응답인 경우 명확한 설명 요청
    if not result.tool_calls and (
        not result.content
        or isinstance(result.content, list)
        and not result.content[0].get("text")
    ):
        messages = state["messages"] + [("user", "유효한 응답을 제공해주세요.")]
        result = assistant.invoke({"messages": messages})

    return {"messages": result}


# 7. 그래프 구성
builder = StateGraph(State)

# 노드 정의: 실제 작업을 수행하는 컴포넌트
builder.add_node("assistant", process_message)
builder.add_node("tools", ToolNode(available_tools))

# 엣지 정의: 노드 간의 데이터 흐름
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# 8. 그래프 컴파일
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(temperature=0, model_name="gpt-4")


# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: State):
    """농담 생성"""

    print("call_llm_1 호출")
    msg = llm.invoke(f"{state['topic']}에 대한 재미있는 농담을 만들어주세요.")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """이야기 생성"""

    print("call_llm_2 호출")
    msg = llm.invoke(f"{state['topic']}에 대한 짧은 이야기를 만들어주세요.")
    return {"story": msg.content}


def call_llm_3(state: State):
    """시 생성"""

    print("call_llm_3 호출")
    msg = llm.invoke(f"{state['topic']}에 대한 짧은 시를 만들어주세요.")
    return {"poem": msg.content}


def aggregator(state: State):
    """모든 결과를 하나로 통합"""

    combined = f"{state['topic']}에 대한 이야기, 농담, 시를 소개합니다!\n\n"
    combined += f"📖 이야기:\n{state['story']}\n\n"
    combined += f"😄 농담:\n{state['joke']}\n\n"
    combined += f"🎵 시:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

graph = parallel_builder.compile()

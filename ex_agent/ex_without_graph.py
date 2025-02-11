from tools import add, multiply, divide
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 툴 통합
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

agent = llm_with_tools.invoke("What is 2 + 2? What is 2 * 2? What is 2 / 2?")

print(agent)

# 테스트
# python ex_agent/ex_agent.py
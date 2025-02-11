from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage


# 커스텀 ChatPromptTemplate 클래스 생성
class DebugChatPromptTemplate(ChatPromptTemplate):
    def format_messages(self, **kwargs) -> list[BaseMessage]:
        messages = super().format_messages(**kwargs)
        # 메시지 내용 출력
        print("\n=== 프롬프트 메시지 ===")
        for msg in messages:
            print(f"[{msg.type}]: {msg.content}")
        print("\n=====================")
        return messages

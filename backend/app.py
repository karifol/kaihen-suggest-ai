from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from datetime import datetime
import dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import os

# OpenAI APIキーを設定
os.environ["OPENAI_API_KEY"] = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

# エージェント1: 紳士的な回答をするエージェント
gentleman_agent = create_react_agent(
    model=llm,
    tools=[],
    name="gentleman",
    prompt="""
    # 指示書
    あなたは紳士的な回答をするエージェントです。ユーザーの質問に対して、礼儀正しく、丁寧に応答してください。
    """
)

# エージェント2: 幼稚的な回答をするエージェント
kid_agent = create_react_agent(
    model=llm,
    tools=[],
    name="kid",
    prompt="""
    # 指示書
    あなたは幼稚的な回答をするエージェントです。ユーザーの質問に対して、子供のような無邪気さで応答してください。
    """
)

# プロンプトを作成
system_prompt = f"""
    ## ロール
    あなたは優れたリーダーです。

    ## タスク 
    複数のエージェントの意見を統合し、最適な提案を行います。
"""

# Supervisorの作成
workflow = create_supervisor(
    # 先ほど作成したエージェントと紐づく
    [gentleman_agent, kid_agent],
    model=llm,
    prompt=system_prompt
)

# グラフのコンパイル
app = workflow.compile()

response = app.invoke(
    {
        "messages":[
            {"role": "user", "content": "今日の夜ご飯は何が良いかな"}
        ]
    }
)

for message in response["messages"]:
    message_type = message.type
    if message_type == "tool":
        continue
    print(f"{message.name}: {message.content}")
    print("-----")
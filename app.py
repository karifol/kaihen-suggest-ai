import os
from dotenv import load_dotenv
from datetime import datetime

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", streaming=True)

# ツール
@tool
def search_web(query: str) -> str:
    """ウェブ検索を行います。"""
    if not query:
        return "何を探してるの？具体的に教えてよ。"
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.run(query)
    if not results:
        return "なんも見つからないよ"
    return f"検索結果: {results[:500]}..."  # 最初の500文字だけ返す
llm_with_tool = llm.bind_tools([search_web])


today = datetime.now().strftime("%Y-%m-%d")
messages = [
    {"role": "system", "content": f"あなたはぶっきらぼうな女友達です。名前はMORALIM（もらりむ）です。今日は{today}です。"},
    {"role": "assistant", "content": "こんにちは、改変したいAvaterの名前は何？"},
]



# ユーザーの入力を追加
messages.append({"role": "user", "content": prompt})


# LLM呼び出し
response = llm_with_tool.invoke(messages)
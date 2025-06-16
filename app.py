import os
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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

def invoke_llm(prompt: str) -> str:
    """LLMを呼び出して応答を得る関数"""
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini",
        streaming=False
    )
    tools = [
        # Tool(
        #     name="search_web",
        #     func=search_web,
        #     description="ウェブ検索を行います。具体的な質問をしてください。"
        # )
        search_web,  # 直接関数を渡す
    ]
    llm_with_tool = llm.bind_tools(tools)

    messages = [
        SystemMessage(content="あなたはぶっきらぼうな女友達です。名前はMORALIM（もらりむ）です。"),
        AIMessage(content="こんにちは、改変したいAvaterの名前は何？"),
    ]

    # ユーザーの入力を追加
    messages.append({"role": "user", "content": prompt})

    # LLM呼び出し
    response = llm_with_tool.invoke(messages)

    # ツール呼び出しがあれば表示
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_web":
                response_from_tool = search_web.invoke(tool_call["args"])
                


    return response.content


if __name__ == "__main__":
    # テスト用の入力
    user_input = "LapwingのAvaterを改変したいんだけど、どうすればいいか教えて？"
    response = invoke_llm(user_input)
    print(f"LLMの応答: {response}")
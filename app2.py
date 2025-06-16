from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
from datetime import datetime
import dotenv
import boto3
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# dynamoDBの設定
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('LangChainChatSessionTable')

# OpenAI APIキーを設定
import os
os.environ["OPENAI_API_KEY"] = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

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

# 状態定義
class ChatState(TypedDict):
    user_input: str
    bot_output: str
    history : list[dict]  # 各メッセージを辞書形式で保持

# OpenAIモデル初期化
llm = ChatOpenAI(model="gpt-4o-mini")  # または gpt-3.5-turbo
llm_with_react = create_react_agent(llm, tools=[search_web])

# ノード1: ユーザー入力をそのまま次へ
def receive_input(state: ChatState) -> ChatState:
    return state

# ノード2: LLMで応答生成
def generate_response(state: ChatState) -> ChatState:
    # response = llm.invoke(state["history"]).content
    response = llm_with_react.invoke({
        "messages": state["history"]
    })
    response = response["messages"][-1].content  # 最後のメッセージを表示
    return {**state, "bot_output": response}

# ノード3: 結果を表示（またはUI連携などに返す）
def return_output(state: ChatState) -> ChatState:
    print(f"🤖 Bot: {state['bot_output']}")
    return state

# LangGraphを構築
builder = StateGraph(ChatState)
builder.add_node("receive", receive_input)
builder.add_node("generate", generate_response)
builder.add_node("output", return_output)

# グラフの流れを定義
builder.set_entry_point("receive")
builder.add_edge("receive", "generate")
builder.add_edge("generate", "output")
builder.add_edge("output", END)

# グラフをコンパイル
graph = builder.compile()

# 初期状態を設定
# dynamobdに会話履歴が存在する場合はそれを取得
if table.get_item(Key={'SessionId': 'example_session_id'}).get('Item'):
    item = table.get_item(Key={'SessionId': 'example_session_id'})['Item']
    history = item['history']
else:
    history = [
        {"role": "system", "content": "あなたはぶっきらぼうな女友達です。名前はMORALIM（もらりむ）です。"},
        {"role": "assistant", "content": "なんか用？"}
    ]

# 画面表示
for message in history:
    if message['role'] == 'user':
        print(f"あなた: {message['content']}")
    elif message['role'] == 'assistant':
        print(f"🤖 Bot: {message['content']}")

# ユーザーからの入力でチャットを開始
user_input = input("あなた: ")

state = {
    "user_input": user_input,
    "bot_output": "",
    "history": history # 初期入力を履歴に追加
}
state["history"].append(
    {"role": "user", "content": user_input}  # 初期応答を履歴に追加
    )  # 初期応答を履歴に追加

# グラフを実行
state = graph.invoke(state)

# 最終的な応答を履歴に追加
state["history"].append(
    {"role": "assistant", "content": state["bot_output"]}
)

# 会話履歴をDynamoDBに保存
session_id = "example_session_id"  # セッションIDを適切に設定
timestamp = datetime.now().isoformat()
table.put_item(
    Item={
        'SessionId': session_id,
        'timestamp': timestamp,
        'history': state['history']
    }
)
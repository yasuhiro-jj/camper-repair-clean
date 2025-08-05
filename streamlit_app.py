import streamlit as st
import os
import uuid
from typing import Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
import glob



# === ページ設定 ===
st.set_page_config(
    page_title="キャンピングカー修理専門AIチャット",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# サイドバーを強制的に表示
st.markdown("""
<style>
/* サイドバーを常に表示 */
.stApp > div[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
}

/* スマホでのサイドバー表示を確保 */
@media (max-width: 768px) {
    .stApp > div[data-testid="stSidebar"] {
        display: block !important;
        width: 100% !important;
        visibility: visible !important;
    }
}
</style>
""", unsafe_allow_html=True)

# === セッション状態の初期化 ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# === データベース初期化 ===
@st.cache_resource
def initialize_database():
    """データベースを初期化"""
    main_path = os.path.dirname(os.path.abspath(__file__))
    
    documents = []
    
    # PDFファイルを動的に検索
    pdf_pattern = os.path.join(main_path, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            pass
    
    # テキストファイルを動的に検索
    txt_pattern = os.path.join(main_path, "*.txt")
    txt_files = glob.glob(txt_pattern)
    
    for txt_path in txt_files:
        try:
            loader = TextLoader(txt_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            pass
    
    if not documents:
        pdf_path = os.path.join(main_path, "キャンピングカー修理マニュアル.pdf")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    
    # OpenAIの埋め込みモデルを設定
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))    
    for doc in documents:
        if not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)
    
    # ドキュメントをメモリに保存
    return documents

# === モデルとツールの設定 ===
@st.cache_resource
def initialize_model():
    """モデルを初期化"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        
        max_tokens=500  # トークン数を制限
    )

@st.cache_resource
def initialize_tools():
    """ツールを初期化"""
    @tool
    def search(query: str):
        """キャンピングカー修理に関する情報を検索します。"""
        try:
            from langchain_community.utilities import SerpAPIWrapper
            
            search_wrapper = SerpAPIWrapper(serpapi_api_key=os.getenv("SERP_API_KEY"))
            result = search_wrapper.run(query)
            
            # 検索結果を箇条書き形式で処理
            if result:
                links = [
                    f"[検索] Google検索: {query} についての詳細情報",
                    f"[動画] YouTube動画: {query} の修理手順動画",
                    f"[購入] Amazon商品: {query} 関連の部品・工具",
                    f"[情報] 専門サイト: キャンピングカー修理専門情報"
                ]
            else:
                links = [
                    f"[検索] Google検索: キャンピングカー {query} 修理方法",
                    f"[動画] YouTube動画: キャンピングカー {query} 修理手順",
                    f"[購入] Amazon商品: キャンピングカー修理部品",
                    f"[情報] 専門サイト: キャンピングカー修理専門情報"
                ]
            
            return links
        except Exception as e:
            return [
                f"[検索] Google検索: キャンピングカー {query} 修理方法",
                f"[動画] YouTube動画: キャンピングカー {query} 修理手順",
                f"[購入] Amazon商品: キャンピングカー修理部品",
                f"[情報] 専門サイト: キャンピングカー修理専門情報"
            ]
    
    return [search]

# === RAGとプロンプトテンプレート ===
def rag_retrieve(question: str, documents):
    """RAGで関連文書を取得"""
    # キーワードベースの検索
    relevant_docs = []
    keywords = question.lower().split()
    
    for doc in documents:
        doc_content = doc.page_content.lower()
        score = sum(1 for keyword in keywords if keyword in doc_content)
        if score > 0:
            relevant_docs.append((doc, score))
    
    # スコアでソート
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_docs:
        content = relevant_docs[0][0].page_content
        if len(content) > 1000:
            content = content[:1000] + "..."
        return content
    else:
        return "キャンピングカーの修理に関する一般的な情報をお探しします。"

template = """
あなたはキャンピングカーの修理専門家です。以下の文書抜粋を参照して質問に答えてください。

文書抜粋：{document_snippet}

質問：{question}

以下の形式で親しみやすい会話調で回答してください：

【対処法】
• 具体的な手順
• 注意点
• 必要な工具・部品

【関連リンク】
• Google検索: {question} 修理方法
• YouTube動画: キャンピングカー {question} 修理手順
• Amazon商品: キャンピングカー修理部品

【岡山キャンピングカー修理サポートセンター】
上記の対処法を試していただき、さらに詳しいアドバイスや実際の修理作業が必要な場合は、お気軽に岡山キャンピングカー修理サポートセンターまでご相談ください！

📞 **お電話でのご相談**
- 電話番号: 080-206-6622
- 営業時間: 年中無休（9:00〜21:00）
- ※不在時は折り返しお電話差し上げます

💬 **メールでのご相談**
- 直接スタッフと相談したいときは直接お電話いただくか、[メールでのご相談](https://camper-repair.net/contact/)も承ります。

**※重要**: 安全な修理作業のため、複雑な修理や専門的な作業が必要な場合は、必ず岡山キャンピングカー修理サポートセンターにご相談ください。

答え：
"""

# === ワークフローの構築 ===
@st.cache_resource
def build_workflow():
    """ワークフローを構築"""
    model = initialize_model()
    tools = initialize_tools()
    tool_node = ToolNode(tools)
    
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END
    
    def call_model(state: MessagesState):
        messages = state['messages']
        try:
            response = model.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            error_message = f"申し訳ございませんが、エラーが発生しました: {str(e)}"
            return {"messages": [AIMessage(content=error_message)]}
    
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", 'agent')
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

# === メインアプリケーション ===
def main():
    # レスポンシブなタイトル（スマホ対応）とヘッダー非表示
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .mobile-title h1 {
            font-size: 1.4rem !important;
            line-height: 1.3 !important;
        }
        .mobile-title p {
            font-size: 0.8rem !important;
        }
    }
    
    /* 右上のメニュー要素を非表示 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ハンバーガーメニューを非表示 */
    .stDeployButton {display: none;}
    
    /* ヘッダー要素を非表示 */
    .stApp > header {display: none;}
    
    /* 右上のツールバー要素を非表示 */
    .stApp > div[data-testid="stToolbar"] {display: none;}
    .stApp > div[data-testid="stToolbarActions"] {display: none;}
    
    /* メニューボタンを非表示 */
    .stApp > div[data-testid="stMenuButton"] {display: none;}
    .stApp > div[data-testid="stMenu"] {display: none;}
    
    /* ヘッダーアクションを非表示 */
    .stApp > div[data-testid="stHeaderActions"] {display: none;}
    
    /* メインコンテンツの上部マージンを調整 */
    .main .block-container {
        padding-top: 1rem;
    }
    
    /* サイドバーを常に表示 */
    .stApp > div[data-testid="stSidebar"] {
        display: block !important;
    }
    
    /* スマホでのサイドバー表示を確保 */
    @media (max-width: 768px) {
        .stApp > div[data-testid="stSidebar"] {
            display: block !important;
            width: 100% !important;
        }
    }
    </style>
    <div class="mobile-title" style="text-align: center;">
        <h1 style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔧 キャンピングカー修理専門AIチャット</h1>
        <p style="font-size: 0.9rem; color: #666; margin-top: 0;">経験豊富なAIがキャンピングカーの修理について詳しくお答えします</p>
    </div>
    """, unsafe_allow_html=True)
    
    # クイック質問をメインエリアに表示（スマホ対応）
    st.markdown("### 📋 クイック質問")
    
    # ボタンを横並びで表示
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔋 バッテリー上がり", use_container_width=True):
            prompt = "バッテリーが上がってエンジンが始動しない時の対処法を教えてください"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
        
        if st.button("🚰 水道ポンプ", use_container_width=True):
            prompt = "水道ポンプから水が出ない時の修理方法は？"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
        
        if st.button("🔥 ガスコンロ", use_container_width=True):
            prompt = "ガスコンロが点火しない時の対処法を教えてください"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
    
    with col2:
        if st.button("🧊 冷蔵庫", use_container_width=True):
            prompt = "冷蔵庫が冷えない時の修理方法は？"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
        
        if st.button("🔧 定期点検", use_container_width=True):
            prompt = "キャンピングカーの定期点検項目とスケジュールは？"
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()
        
        if st.button("🆕 新しい会話", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
    
    st.divider()
    
    # メインエリア
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # クイック質問からの自動回答処理
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # 最新のメッセージがユーザーからの場合、AI回答を生成
        prompt = st.session_state.messages[-1]["content"]
        
        # AIの回答を生成
        with st.chat_message("assistant", avatar="https://camper-repair.net/blog/wp-content/uploads/2025/05/dummy_staff_01-150x138-1.png"):
            with st.spinner("🔧 修理アドバイスを生成中..."):
                try:
                    # ドキュメントとワークフローを取得
                    documents = initialize_database()
                    app_flow = build_workflow()
                    
                    # RAGで関連文書を取得
                    document_snippet = rag_retrieve(prompt, documents)
                    
                    # プロンプトを構築
                    content = template.format(document_snippet=document_snippet, question=prompt)
                    
                    # 会話履歴を構築（最新の5件のみ）
                    history = []
                    recent_messages = st.session_state.messages[-5:-1]  # 最新の5件のみ
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            history.append(HumanMessage(content=msg["content"]))
                        else:
                            history.append(AIMessage(content=msg["content"]))
                    
                    # 新しいメッセージを追加
                    inputs = history + [HumanMessage(content=content)]
                    thread = {"configurable": {"thread_id": st.session_state.conversation_id}}
                    
                    # 回答を生成
                    response = ""
                    for event in app_flow.stream({"messages": inputs}, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            response = event["messages"][-1].content
                    
                    # 回答を表示
                    st.markdown(response)
                    
                    # 関連リンクを表示
                    st.markdown("---")
                    st.markdown("**🔗 関連リンク**")
                    
                    # Google検索リンク
                    google_query = f"キャンピングカー {prompt} 修理方法"
                    google_url = f"https://www.google.com/search?q={google_query.replace(' ', '+')}"
                    st.markdown(f"🔍 **[Google検索: {prompt}の修理方法]({google_url})**")
                    st.markdown(f"*詳細な修理手順、専門業者情報、トラブルシューティング方法を検索*")
                    
                    # YouTube検索リンク
                    youtube_query = f"キャンピングカー {prompt} 修理"
                    youtube_url = f"https://www.youtube.com/results?search_query={youtube_query.replace(' ', '+')}"
                    st.markdown(f"📺 **[YouTube動画: {prompt}の修理手順]({youtube_url})**")
                    st.markdown(f"*実際の修理作業の動画、工具の使い方、部品交換の手順を視聴*")
                    
                    # Amazon検索リンク
                    amazon_query = f"キャンピングカー 修理 部品"
                    amazon_url = f"https://www.amazon.co.jp/s?k={amazon_query.replace(' ', '+')}"
                    st.markdown(f"🛒 **[Amazon商品: キャンピングカー修理部品]({amazon_url})**")
                    st.markdown(f"*必要な工具、交換部品、消耗品の購入*")
                    
                    # 岡山サポートセンターリンク
                    st.markdown("---")
                    st.markdown("**🏢 岡山キャンピングカー修理サポートセンター**")
                    st.markdown("📞 **電話番号**: 080-206-6622")
                    st.markdown("⏰ **営業時間**: 年中無休（9:00〜21:00）")
                    st.markdown("*※不在時は折り返しお電話差し上げます*")
                    st.markdown("💬 **メールでのご相談**: [お問合わせフォーム](https://camper-repair.net/contact/)")
                    st.markdown("*直接スタッフと相談したいときは直接お電話いただくか、メールでのご相談も承ります*")
                    st.markdown("**⚠️ 重要**: 安全な修理作業のため、複雑な修理や専門的な作業が必要な場合は、必ず岡山キャンピングカー修理サポートセンターにご相談ください。")
                    
                    # AIメッセージを履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
    
    # ユーザー入力（常に最後に表示）
    if prompt := st.chat_input("キャンピングカーの修理について質問してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AIの回答を生成
        with st.chat_message("assistant", avatar="https://camper-repair.net/blog/wp-content/uploads/2025/05/dummy_staff_01-150x138-1.png"):
            with st.spinner("🔧 修理アドバイスを生成中..."):
                try:
                    # ドキュメントとワークフローを取得
                    documents = initialize_database()
                    app_flow = build_workflow()
                    
                    # RAGで関連文書を取得
                    document_snippet = rag_retrieve(prompt, documents)
                    
                    # プロンプトを構築
                    content = template.format(document_snippet=document_snippet, question=prompt)
                    
                    # 会話履歴を構築（最新の5件のみ）
                    history = []
                    recent_messages = st.session_state.messages[-5:-1]  # 最新の5件のみ
                    for msg in recent_messages:
                        if msg["role"] == "user":
                            history.append(HumanMessage(content=msg["content"]))
                        else:
                            history.append(AIMessage(content=msg["content"]))
                    
                    # 新しいメッセージを追加
                    inputs = history + [HumanMessage(content=content)]
                    thread = {"configurable": {"thread_id": st.session_state.conversation_id}}
                    
                    # 回答を生成
                    response = ""
                    for event in app_flow.stream({"messages": inputs}, thread, stream_mode="values"):
                        if "messages" in event and event["messages"]:
                            response = event["messages"][-1].content
                    
                    # 回答を表示
                    st.markdown(response)
                    
                    # 関連リンクを表示
                    st.markdown("---")
                    st.markdown("**🔗 関連リンク**")
                    
                    # Google検索リンク
                    google_query = f"キャンピングカー {prompt} 修理方法"
                    google_url = f"https://www.google.com/search?q={google_query.replace(' ', '+')}"
                    st.markdown(f"🔍 **[Google検索: {prompt}の修理方法]({google_url})**")
                    st.markdown(f"*詳細な修理手順、専門業者情報、トラブルシューティング方法を検索*")
                    
                    # YouTube検索リンク
                    youtube_query = f"キャンピングカー {prompt} 修理"
                    youtube_url = f"https://www.youtube.com/results?search_query={youtube_query.replace(' ', '+')}"
                    st.markdown(f"📺 **[YouTube動画: {prompt}の修理手順]({youtube_url})**")
                    st.markdown(f"*実際の修理作業の動画、工具の使い方、部品交換の手順を視聴*")
                    
                    # Amazon検索リンク
                    amazon_query = f"キャンピングカー 修理 部品"
                    amazon_url = f"https://www.amazon.co.jp/s?k={amazon_query.replace(' ', '+')}"
                    st.markdown(f"🛒 **[Amazon商品: キャンピングカー修理部品]({amazon_url})**")
                    st.markdown(f"*必要な工具、交換部品、消耗品の購入*")
                    
                    # 岡山サポートセンターリンク
                    st.markdown("---")
                    st.markdown("**🏢 岡山キャンピングカー修理サポートセンター**")
                    st.markdown("📞 **電話番号**: 080-206-6622")
                    st.markdown("⏰ **営業時間**: 年中無休（9:00〜21:00）")
                    st.markdown("*※不在時は折り返しお電話差し上げます*")
                    st.markdown("💬 **メールでのご相談**: [お問合わせフォーム](https://camper-repair.net/contact/)")
                    st.markdown("*直接スタッフと相談したいときは直接お電話いただくか、メールでのご相談も承ります*")
                    st.markdown("**⚠️ 重要**: 安全な修理作業のため、複雑な修理や専門的な作業が必要な場合は、必ず岡山キャンピングカー修理サポートセンターにご相談ください。")
                    
                    # AIメッセージを履歴に追加
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 

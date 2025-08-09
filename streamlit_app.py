import streamlit as st
import os
import uuid
import re
from typing import Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import glob
import config

# === ブログURL抽出関数 ===
def extract_blog_urls(documents, question=""):
    """文書からブログURLを抽出"""
    urls = set()
    
    for doc in documents:
        content = doc.page_content
        # URLパターンを検索（https://camper-repair.net/で始まるURL）
        url_pattern = r'https://camper-repair\.net/[^\s,、，。]+'
        found_urls = re.findall(url_pattern, content)
        urls.update(found_urls)
    
    # 質問に関連するURLを優先的に表示
    if question:
        relevant_urls = []
        other_urls = []
        
        for url in urls:
            if any(keyword in url.lower() for keyword in question.lower().split()):
                relevant_urls.append(url)
            else:
                other_urls.append(url)
        
        # 関連URLを先頭に、その他を後ろに
        return list(relevant_urls) + list(other_urls)
    
    return list(urls)

def extract_scenario_related_blogs(documents, question=""):
    """シナリオファイルから関連ブログを抽出"""
    related_blogs = []
    
    # 質問のキーワードを抽出
    keywords = question.lower().split()
    
    for doc in documents:
        content = doc.page_content.lower()
        source = doc.metadata.get('source', '')
        
        # テキストファイル（シナリオ）の場合
        if source.endswith('.txt'):
            # キーワードに基づいて関連性を判定
            relevance_score = 0
            
            # 冷蔵庫関連
            if any(word in content for word in ['冷蔵庫', 'refrigerator', '冷蔵', '庫内', 'コンプレッサ']):
                if any(word in question.lower() for word in ['冷蔵庫', '冷蔵', '冷えない', '冷凍']):
                    relevance_score += 10
            
            # FFヒーター関連
            if any(word in content for word in ['ffヒーター', 'ff', 'ヒーター', '暖房']):
                if any(word in question.lower() for word in ['ff', 'ヒーター', '暖房', '暖かい']):
                    relevance_score += 10
            
            # 雨漏り関連
            if any(word in content for word in ['雨漏り', '雨', '漏水', '水漏れ']):
                if any(word in question.lower() for word in ['雨漏り', '雨', '漏水', '水漏れ']):
                    relevance_score += 10
            
            # バッテリー関連
            if any(word in content for word in ['バッテリー', 'battery', '電源', '充電']):
                if any(word in question.lower() for word in ['バッテリー', '電源', '充電', '上がり']):
                    relevance_score += 10
            
            # 水道ポンプ関連
            if any(word in content for word in ['水道ポンプ', '水', 'ポンプ', '給水']):
                if any(word in question.lower() for word in ['水道', '水', 'ポンプ', '給水']):
                    relevance_score += 10
            
            # ガスコンロ関連
            if any(word in content for word in ['ガスコンロ', 'ガス', 'コンロ', '点火']):
                if any(word in question.lower() for word in ['ガス', 'コンロ', '点火', '火']):
                    relevance_score += 10
            
            # トイレ関連
            if any(word in content for word in ['トイレ', 'toilet', '便器', '排水']):
                if any(word in question.lower() for word in ['トイレ', '便器', '排水']):
                    relevance_score += 10
            
            # ソーラーパネル関連
            if any(word in content for word in ['ソーラーパネル', 'solar', '太陽光', '発電']):
                if any(word in question.lower() for word in ['ソーラー', '太陽光', '発電']):
                    relevance_score += 10
            
            # インバーター関連
            if any(word in content for word in ['インバーター', 'inverter', '変換器', 'ac']):
                if any(word in question.lower() for word in ['インバーター', '変換器', 'ac', '交流']):
                    relevance_score += 10
            
            # 家具関連
            if any(word in content for word in ['家具', 'テーブル', '椅子', 'ベッド']):
                if any(word in question.lower() for word in ['家具', 'テーブル', '椅子', 'ベッド']):
                    relevance_score += 10
            
            # 換気扇関連
            if any(word in content for word in ['換気扇', 'vent', '換気', 'ファン']):
                if any(word in question.lower() for word in ['換気扇', '換気', 'ファン']):
                    relevance_score += 10
            
            # 窓関連
            if any(word in content for word in ['窓', 'window', 'ガラス', 'サッシ']):
                if any(word in question.lower() for word in ['窓', 'ガラス', 'サッシ']):
                    relevance_score += 10
            
            # 車体外装関連
            if any(word in content for word in ['車体', '外装', 'ボディ', '塗装']):
                if any(word in question.lower() for word in ['車体', '外装', 'ボディ', '塗装']):
                    relevance_score += 10
            
            # 異音関連
            if any(word in content for word in ['異音', '音', '騒音', '振動']):
                if any(word in question.lower() for word in ['異音', '音', '騒音', '振動']):
                    relevance_score += 10
            
            # 関連性が高い場合、シナリオファイル名からブログ情報を生成
            if relevance_score > 0:
                # ファイル名からタイトルを抽出
                filename = os.path.basename(source)
                if '冷蔵庫' in filename:
                    blog_info = {
                        'title': '冷蔵庫の故障と修理方法',
                        'url': 'https://camper-repair.net/refrigerator-repair',
                        'category': '🧊 冷蔵庫',
                        'relevance_score': relevance_score
                    }
                elif 'ff' in filename.lower() or 'ヒーター' in filename:
                    blog_info = {
                        'title': 'FFヒーターの故障と修理方法',
                        'url': 'https://camper-repair.net/ff-heater-repair',
                        'category': '🔥 FFヒーター',
                        'relevance_score': relevance_score
                    }
                elif '雨漏り' in filename:
                    blog_info = {
                        'title': '雨漏りの対処法と修理',
                        'url': 'https://camper-repair.net/rain-leak-repair',
                        'category': '🌧️ 雨漏り',
                        'relevance_score': relevance_score
                    }
                elif 'バッテリー' in filename:
                    blog_info = {
                        'title': 'バッテリーの故障と交換方法',
                        'url': 'https://camper-repair.net/battery-repair',
                        'category': '🔋 バッテリー',
                        'relevance_score': relevance_score
                    }
                elif '水道' in filename or 'ポンプ' in filename:
                    blog_info = {
                        'title': '水道ポンプの修理方法',
                        'url': 'https://camper-repair.net/water-pump-repair',
                        'category': '🚰 水道ポンプ',
                        'relevance_score': relevance_score
                    }
                elif 'ガス' in filename:
                    blog_info = {
                        'title': 'ガスコンロの点火トラブル対処',
                        'url': 'https://camper-repair.net/gas-stove-repair',
                        'category': '🔥 ガスコンロ',
                        'relevance_score': relevance_score
                    }
                elif 'トイレ' in filename:
                    blog_info = {
                        'title': 'トイレの故障と修理方法',
                        'url': 'https://camper-repair.net/toilet-repair',
                        'category': '🚽 トイレ',
                        'relevance_score': relevance_score
                    }
                elif 'ソーラー' in filename or 'solar' in filename.lower():
                    blog_info = {
                        'title': 'ソーラーパネルの設置と修理',
                        'url': 'https://camper-repair.net/solar-panel-repair',
                        'category': '☀️ ソーラーパネル',
                        'relevance_score': relevance_score
                    }
                elif 'インバーター' in filename or 'inverter' in filename.lower():
                    blog_info = {
                        'title': 'インバーターの故障と修理',
                        'url': 'https://camper-repair.net/inverter-repair',
                        'category': '⚡ インバーター',
                        'relevance_score': relevance_score
                    }
                elif '家具' in filename:
                    blog_info = {
                        'title': '家具の修理とメンテナンス',
                        'url': 'https://camper-repair.net/furniture-repair',
                        'category': '🪑 家具',
                        'relevance_score': relevance_score
                    }
                elif '換気' in filename or 'ベント' in filename:
                    blog_info = {
                        'title': '換気扇の故障と修理',
                        'url': 'https://camper-repair.net/vent-repair',
                        'category': '💨 換気扇',
                        'relevance_score': relevance_score
                    }
                elif '窓' in filename or 'window' in filename.lower():
                    blog_info = {
                        'title': '窓の修理と交換方法',
                        'url': 'https://camper-repair.net/window-repair',
                        'category': '🪟 窓',
                        'relevance_score': relevance_score
                    }
                elif '車体' in filename or '外装' in filename:
                    blog_info = {
                        'title': '車体外装の修理方法',
                        'url': 'https://camper-repair.net/exterior-repair',
                        'category': '🚗 車体外装',
                        'relevance_score': relevance_score
                    }
                elif '異音' in filename:
                    blog_info = {
                        'title': '異音の原因と対処法',
                        'url': 'https://camper-repair.net/noise-repair',
                        'category': '🔊 異音',
                        'relevance_score': relevance_score
                    }
                else:
                    # その他のシナリオファイル
                    blog_info = {
                        'title': f'{filename.replace(".txt", "").replace("シナリオ", "").strip()}の修理方法',
                        'url': 'https://camper-repair.net/general-repair',
                        'category': '🔧 修理全般',
                        'relevance_score': relevance_score
                    }
                
                related_blogs.append(blog_info)
    
    # 関連性スコアでソート（高い順）
    related_blogs.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return related_blogs

def extract_title_from_url(url):
    """URLから適切なタイトルを抽出"""
    # URLのパス部分を取得
    path = url.split('/')
    
    # 最後の部分（ファイル名）を取得
    filename = path[-1] if path[-1] else path[-2] if len(path) > 1 else ""
    
    # ファイル名から拡張子を除去
    if '.' in filename:
        filename = filename.split('.')[0]
    
    # ハイフンやアンダースコアをスペースに変換
    title = filename.replace('-', ' ').replace('_', ' ')
    
    # カテゴリ別のタイトルマッピング
    title_mapping = {
        'ff': 'FFヒーターの修理方法',
        'rain': '雨漏りの対処法と修理',
        'inverter': 'インバーターの故障と修理',
        'electrical': '電気系統のトラブル対処',
        'battery': 'バッテリーの故障と交換',
        'water': '水道ポンプの修理方法',
        'gas': 'ガスコンロの点火トラブル',
        'refrigerator': '冷蔵庫の故障と修理',
        'toilet': 'トイレの故障と修理',
        'solar': 'ソーラーパネルの設置と修理',
        'furniture': '家具の修理とメンテナンス',
        'vent': '換気扇の故障と修理',
        'window': '窓の修理と交換',
        'exterior': '車体外装の修理',
        'noise': '異音の原因と対処法'
    }
    
    # キーワードに基づいてタイトルを決定
    for keyword, mapped_title in title_mapping.items():
        if keyword in url.lower():
            return mapped_title
    
    # デフォルトのタイトル生成
    if title:
        # 各単語の最初の文字を大文字に
        title = ' '.join(word.capitalize() for word in title.split())
        return f"{title}の修理方法"
    
    return "キャンピングカー修理情報"

def categorize_blog_urls(urls):
    """ブログURLをカテゴリ別に分類"""
    categories = {
        "FFヒーター": [],
        "雨漏り": [],
        "外部電源": [],
        "その他": []
    }
    
    for url in urls:
        if "ff" in url.lower():
            categories["FFヒーター"].append(url)
        elif "rain" in url.lower():
            categories["雨漏り"].append(url)
        elif "inverter" in url.lower() or "electrical" in url.lower():
            categories["外部電源"].append(url)
        else:
            categories["その他"].append(url)
    
    return categories

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
    
    # ドキュメントの内容を文字列に変換
    for doc in documents:
        if not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)
    
    # ドキュメントをメモリに保存
    return documents

# === モデルとツールの設定 ===
@st.cache_resource
def initialize_model():
    """モデルを初期化"""
    # APIキーを環境変数から取得
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 環境変数が設定されていない場合の処理
    if not api_key:
        st.error("⚠️ OpenAI APIキーが設定されていません。")
        st.info("セキュリティのため、以下の方法でAPIキーを設定してください：")
        st.markdown("""
        **推奨方法（セキュリティ重視）**:
        
        1. **環境変数ファイル**: `env_example.txt`を参考に`.env`ファイルを作成
        2. **システム環境変数**: Windowsの環境変数に`OPENAI_API_KEY`を追加
           
        **⚠️ セキュリティ注意事項**:
        - APIキーをコード内に直接記述しないでください
        - `.env`ファイルをGitにコミットしないでください
        - APIキーを公開リポジトリにアップロードしないでください
        """)
        return None
    
    return ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500  # トークン数を制限
    )

@st.cache_resource
def initialize_tools():
    """ツールを初期化"""
    # ツールを無効化して関連リンクを表示しない
    return []

# === RAGとプロンプトテンプレート ===
def rag_retrieve(question: str, documents):
    """RAGで関連文書を取得"""
    # キーワードベースの検索
    relevant_docs = []
    keywords = question.lower().split()
    
    # より詳細なキーワード抽出
    important_keywords = []
    for keyword in keywords:
        if len(keyword) > 2:  # 2文字以上のキーワードのみ
            important_keywords.append(keyword)
    
    for doc in documents:
        doc_content = doc.page_content.lower()
        score = 0
        
        # 完全一致の重みを高く
        for keyword in important_keywords:
            if keyword in doc_content:
                score += 2
            # 部分一致も考慮
            if any(keyword in word for word in doc_content.split()):
                score += 1
        
        if score > 0:
            relevant_docs.append((doc, score))
    
    # スコアでソート
    relevant_docs.sort(key=lambda x: x[1], reverse=True)
    
    if relevant_docs:
        # 上位3件の文書を結合
        top_docs = relevant_docs[:3]
        combined_content = ""
        for doc, score in top_docs:
            content = doc.page_content
            if len(content) > 500:  # 各文書を500文字に制限
                content = content[:500] + "..."
            combined_content += f"\n\n---\n{content}"
        
        if len(combined_content) > 1500:
            combined_content = combined_content[:1500] + "..."
        
        return combined_content
    else:
        return "キャンピングカーの修理に関する一般的な情報をお探しします。"

template = """
あなたはキャンピングカーの修理専門家です。以下の文書抜粋を参照して質問に答えてください。

文書抜粋：{document_snippet}

質問：{question}

以下の形式で親しみやすい会話調で回答してください。絶対にリンク、URL、検索結果、動画情報、商品情報、関連リンク、Google検索、YouTube動画、Amazon商品、🔗、🔍、📺、🛒、🏢、📖、📞、🔄、❓、💬、🔧、📋、🆕、🔋、🚰、🔥、🧊、🔧、🆕、【関連リンク】、【関連情報】、【詳細情報】、【参考リンク】、【外部リンク】、【検索結果】、【動画情報】、【商品情報】は含めないでください：

【対処法】
• 具体的な手順
• 注意点
• 必要な工具・部品

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

# === ヘルパー関数 ===
# 関連リンクの表示を無効化
# def display_related_links(prompt: str):
#     """関連ブログのリンクを表示する関数"""
#     st.markdown("---")
#     st.markdown("**🔗 関連ブログ記事**")
#     
#     # キーワードに基づいて関連ブログを検索
#     keywords = prompt.lower().split()
#     blog_links = []
#     
#     # ブログリンクを表示（架空のURLを削除）
#     st.markdown("📖 **キャンピングカー修理の基本知識**")
#     st.markdown("*修理作業の基礎と安全な作業方法*")

def generate_ai_response(prompt: str):
    """AI回答を生成する関数"""
    try:
        # ドキュメントとワークフローを取得
        documents = initialize_database()
        app_flow = build_workflow()
        
        # RAGで関連文書を取得
        document_snippet = rag_retrieve(prompt, documents)
        
        # プロンプトを構築（外部リンクを完全に除外）
        content = template.format(document_snippet=document_snippet, question=prompt) + "\n\n重要：回答には絶対に外部リンク、URL、関連リンク、【関連リンク】、【関連情報】、【詳細情報】、【参考リンク】、【外部リンク】、【検索結果】、【動画情報】、【商品情報】、🔗、🔍、📺、🛒、🏢、📖、📞、🔄、❓、💬、🔧、📋、🆕、🔋、🚰、🔥、🧊、🔧、🆕、Google検索、YouTube動画、Amazon商品、• Google検索、• YouTube動画、• Amazon商品を含めないでください。純粋な修理アドバイスのみを提供してください。【対処法】セクションのみを含めてください。⚠️ 重要: 安全な修理作業のため、複雑な修理や専門的な作業が必要な場合は、岡山キャンピングカー修理サポートセンターにご相談ください。"
        
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
        
        # デバッグ用：元の回答を確認
        print("Original response:", response)
        
        # 回答からリンクを除去して表示
        
        # すべてのURLを除去
        clean_response = re.sub(r'https?://[^\s]+', '', response)
        
        # すべてのMarkdownリンクを除去
        clean_response = re.sub(r'\[.*?\]\(.*?\)', '', clean_response)
        
        # 関連リンクセクション全体を除去
        clean_response = re.sub(r'【関連リンク】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【関連情報】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【詳細情報】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【参考リンク】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【外部リンク】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【検索結果】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【動画情報】.*?【', '【', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【商品情報】.*?【', '【', clean_response, flags=re.DOTALL)
        
        # リンク関連のアイコンとテキストを除去
        clean_response = re.sub(r'🔗.*?関連.*?🔗', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'🔍.*?検索.*?🔍', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'📺.*?動画.*?📺', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'🛒.*?商品.*?🛒', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'📖.*?情報.*?📖', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'📞.*?サポート.*?📞', '', clean_response, flags=re.DOTALL)
        
        # 具体的なリンクパターンを除去
        clean_response = re.sub(r'• Google検索:.*?$', '', clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r'• YouTube動画:.*?$', '', clean_response, flags=re.MULTILINE)
        clean_response = re.sub(r'• Amazon商品:.*?$', '', clean_response, flags=re.MULTILINE)
        
        # リンク全体を除去
        clean_response = re.sub(r'【関連リンク】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【関連情報】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【詳細情報】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【参考リンク】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【外部リンク】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【検索結果】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【動画情報】.*?$', '', clean_response, flags=re.DOTALL)
        clean_response = re.sub(r'【商品情報】.*?$', '', clean_response, flags=re.DOTALL)
        
        # 空行を整理
        clean_response = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_response)
        
        # 最終的なフィルタリング - 関連リンクセクションが残っている場合は除去
        if '【関連リンク】' in clean_response:
            clean_response = clean_response.split('【関連リンク】')[0]
        if '【関連情報】' in clean_response:
            clean_response = clean_response.split('【関連情報】')[0]
        if '【詳細情報】' in clean_response:
            clean_response = clean_response.split('【詳細情報】')[0]
        if '【参考リンク】' in clean_response:
            clean_response = clean_response.split('【参考リンク】')[0]
        if '【外部リンク】' in clean_response:
            clean_response = clean_response.split('【外部リンク】')[0]
        if '【検索結果】' in clean_response:
            clean_response = clean_response.split('【検索結果】')[0]
        if '【動画情報】' in clean_response:
            clean_response = clean_response.split('【動画情報】')[0]
        if '【商品情報】' in clean_response:
            clean_response = clean_response.split('【商品情報】')[0]
        
        if '🔗 関連リンク' in clean_response:
            clean_response = clean_response.split('🔗 関連リンク')[0]
        
        # 最後の改行を整理
        clean_response = clean_response.strip()
        
        # お問い合わせ案内を追加
        contact_info = "\n\n---\n\n**💬 追加の質問**\n他に何かご質問ありましたら、引き続きチャットボットに聞いてみてください。\n\n**📞 お問い合わせ**\n直接スタッフにお尋ねをご希望の方は、[お問い合わせフォーム](https://camper-repair.net/contact/)またはお電話（086-206-6622）で受付けております。\n\n【営業時間】年中無休（9:00～21:00）\n※不在時は折り返しお電話差し上げます。"
        clean_response += contact_info
        
        # デバッグ用：フィルタリング後の回答を確認
        print("Filtered response:", clean_response)
        
        st.markdown(clean_response)
        
        # 関連ブログを表示
        st.markdown("---")
        st.markdown("**🔗 関連ブログ記事**")
        
        # シナリオファイルから関連ブログを抽出
        scenario_blogs = extract_scenario_related_blogs(documents, prompt)
        
        # URLからも関連ブログを抽出
        url_blogs = extract_blog_urls(documents, prompt)
        
        # 両方の結果を統合
        all_blogs = []
        
        # シナリオ関連ブログを追加
        for blog in scenario_blogs:
            all_blogs.append({
                'title': blog['title'],
                'url': blog['url'],
                'category': blog['category'],
                'source': 'scenario'
            })
        
        # URL関連ブログを追加
        for url in url_blogs[:2]:  # URLは最大2件まで
            title = extract_title_from_url(url)
            category = ""
            if "ff" in url.lower():
                category = "🔥 FFヒーター"
            elif "rain" in url.lower():
                category = "🌧️ 雨漏り"
            elif "inverter" in url.lower() or "electrical" in url.lower():
                category = "⚡ 外部電源"
            else:
                category = "🔧 修理全般"
            
            all_blogs.append({
                'title': title,
                'url': url,
                'category': category,
                'source': 'url'
            })
        
        if all_blogs:
            # 上位3件の関連ブログを表示（関連性の高い順）
            for i, blog in enumerate(all_blogs[:3]):
                # ブログ記事をカード形式で表示（リンクとして機能）
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.markdown(f"**{blog['category']}**")
                    with col2:
                        # 直接リンクとして表示
                        st.markdown(f"[📖 {blog['title']}]({blog['url']})")
                        st.caption(f"カテゴリ: {blog['category']}")
        else:
            st.info("関連するブログ記事が見つかりませんでした")
        
        # 関連リンクの表示を無効化
        # display_related_links(prompt)
        
        # AIメッセージを履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

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
    
    # クイック質問からの自動回答処理
    if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
        # 最新のメッセージがユーザーからの場合、AI回答を生成
        prompt = st.session_state.messages[-1]["content"]
        st.session_state.current_question = prompt  # 現在の質問を保存
        
        # AIの回答を生成
        with st.chat_message("assistant", avatar="https://camper-repair.net/blog/wp-content/uploads/2025/05/dummy_staff_01-150x138-1.png"):
            with st.spinner("🔧 修理アドバイスを生成中..."):
                generate_ai_response(prompt)
    
    # メインエリア
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ユーザー入力（常に最後に表示）
    if prompt := st.chat_input("キャンピングカーの修理について質問してください..."):
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.current_question = prompt  # 現在の質問を保存
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AIの回答を生成
        with st.chat_message("assistant", avatar="https://camper-repair.net/blog/wp-content/uploads/2025/05/dummy_staff_01-150x138-1.png"):
            with st.spinner("🔧 修理アドバイスを生成中..."):
                generate_ai_response(prompt)

if __name__ == "__main__":
    main() 

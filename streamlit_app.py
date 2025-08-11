import streamlit as st
import os
import uuid
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage

# Windows互換性のため、個別にインポート
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ModuleNotFoundError as e:
    if "pwd" in str(e):
        # pwdモジュールエラーの場合、代替手段を使用
        import sys
        import platform
        if platform.system() == "Windows":
            # Windows環境での代替インポート
            from langchain_community.document_loaders.pdf import PyPDFLoader
            from langchain_community.document_loaders.text import TextLoader
        else:
            raise e
    else:
        raise e

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
    """シナリオファイルから関連ブログを抽出（改善版）"""
    related_blogs = []
    
    if not question:
        return []
    
    # 質問のキーワードを抽出
    question_lower = question.lower()
    
    # 実際のファイルからURLを抽出
    actual_urls = {}
    for doc in documents:
        content = doc.page_content
        source = doc.metadata.get('source', '')
        
        # URLパターンを検索（https://camper-repair.net/で始まるURL）
        url_pattern = r'https://camper-repair\.net/[^\s,、，。]+'
        found_urls = re.findall(url_pattern, content)
        
        if found_urls:
            # ファイル名からカテゴリを特定
            filename = os.path.basename(source)
            if '水道ポンプ' in filename:
                actual_urls['水道ポンプ'] = found_urls[0]
            elif '冷蔵庫' in filename:
                actual_urls['冷蔵庫'] = found_urls[0]
            elif 'ffヒーター' in filename.lower() or 'ffヒーター' in filename:
                actual_urls['ffヒーター'] = found_urls[0]
            elif '雨漏り' in filename:
                actual_urls['雨漏り'] = found_urls[0]
            elif 'バッテリー' in filename:
                actual_urls['バッテリー'] = found_urls[0]
            elif 'ガスコンロ' in filename:
                actual_urls['ガスコンロ'] = found_urls[0]
            elif 'トイレ' in filename:
                actual_urls['トイレ'] = found_urls[0]
            elif 'ソーラーパネル' in filename:
                actual_urls['ソーラーパネル'] = found_urls[0]
            elif 'インバーター' in filename:
                actual_urls['インバーター'] = found_urls[0]
            elif '電装系' in filename:
                actual_urls['電装系'] = found_urls[0]
            elif 'ルーフベント' in filename:
                actual_urls['ルーフベント'] = found_urls[0]
            elif '家具' in filename:
                actual_urls['家具'] = found_urls[0]
            elif '外部電源' in filename:
                actual_urls['外部電源'] = found_urls[0]
            elif '排水タンク' in filename:
                actual_urls['排水タンク'] = found_urls[0]
            elif 'ウインドウ' in filename:
                actual_urls['ウインドウ'] = found_urls[0]
            elif '車体外装' in filename:
                actual_urls['車体外装'] = found_urls[0]
            elif '異音' in filename:
                actual_urls['異音'] = found_urls[0]
    
    # 正確なURLとタイトルのマッピング（実際のファイル内容に基づく）
    keyword_mapping = {
        '冷蔵庫': {
            'keywords': ['冷蔵庫', '冷蔵', '冷凍', '冷えない', 'コンプレッサ'],
            'url': actual_urls.get('冷蔵庫', 'https://camper-repair.net/refrigerator/'),
            'title': '冷蔵庫トラブル知識ベース（キャンピングカー用・コンプレッサ式／3WAY共通）',
            'category': '🧊 冷蔵庫'
        },
        'ffヒーター': {
            'keywords': ['ffヒーター', 'ff', 'ヒーター', '暖房', '暖かい', '温風'],
            'url': actual_urls.get('ffヒーター', 'https://camper-repair.net/ff-heater/'),
            'title': 'FFヒーターの故障と修理方法',
            'category': '🔥 FFヒーター'
        },
        '雨漏り': {
            'keywords': ['雨漏り', '雨', '漏水', '水漏れ', '湿気', '防水'],
            'url': actual_urls.get('雨漏り', 'https://camper-repair.net/rain-leak/'),
            'title': '雨漏りの対処法と修理',
            'category': '🌧️ 雨漏り'
        },
        'バッテリー': {
            'keywords': ['バッテリー', 'battery', '電源', '充電', '上がり', '電圧'],
            'url': actual_urls.get('バッテリー', 'https://camper-repair.net/battery/'),
            'title': 'バッテリーの故障と修理方法',
            'category': '🔋 バッテリー'
        },
        '水道ポンプ': {
            'keywords': ['水道ポンプ', '水', 'ポンプ', '給水', '水圧', '蛇口'],
            'url': actual_urls.get('水道ポンプ', 'https://camper-repair.net/water1/'),
            'title': '水道ポンプの故障と修理方法',
            'category': '💧 水道ポンプ'
        },
        'ガスコンロ': {
            'keywords': ['ガスコンロ', 'ガス', 'コンロ', '点火', '火', '燃焼'],
            'url': actual_urls.get('ガスコンロ', 'https://camper-repair.net/gas-stove/'),
            'title': 'ガスコンロの故障と修理方法',
            'category': '🔥 ガスコンロ'
        },
        'トイレ': {
            'keywords': ['トイレ', 'toilet', '便器', '排水', '水洗', '臭い'],
            'url': actual_urls.get('トイレ', 'https://camper-repair.net/toilet/'),
            'title': 'トイレの故障と修理方法',
            'category': '🚽 トイレ'
        },
        'ソーラーパネル': {
            'keywords': ['ソーラーパネル', 'solar', '太陽光', '発電', '充電', 'パネル'],
            'url': actual_urls.get('ソーラーパネル', 'https://camper-repair.net/solar-panel/'),
            'title': 'ソーラーパネルの故障と修理方法',
            'category': '☀️ ソーラーパネル'
        },
        'インバーター': {
            'keywords': ['インバーター', 'inverter', '交流', '直流', '変換', '電圧'],
            'url': actual_urls.get('インバーター', 'https://camper-repair.net/blog/inverter1/'),
            'title': 'インバーター選定と設置方法',
            'category': '⚡ インバーター'
        },
        '電装系': {
            'keywords': ['電装', '配線', '電気', 'ショート', '断線', '電圧'],
            'url': actual_urls.get('電装系', 'https://camper-repair.net/blog/electrical-solar-panel/'),
            'title': 'キャンピングカー配線の基本と電装システム',
            'category': '🔌 電装系'
        },
        'ルーフベント': {
            'keywords': ['ルーフベント', '換気扇', '換気', '空気', '風通し'],
            'url': actual_urls.get('ルーフベント', 'https://camper-repair.net/roof-vent/'),
            'title': 'ルーフベント・換気扇の故障と修理方法',
            'category': '💨 ルーフベント'
        },
        '家具': {
            'keywords': ['家具', 'テーブル', '椅子', 'ベッド', '収納', '破損'],
            'url': actual_urls.get('家具', 'https://camper-repair.net/furniture/'),
            'title': '家具の故障と修理方法',
            'category': '🪑 家具'
        },
        '外部電源': {
            'keywords': ['外部電源', 'コンセント', 'ac', '交流', '充電'],
            'url': actual_urls.get('外部電源', 'https://camper-repair.net/external-power/'),
            'title': '外部電源の故障と修理方法',
            'category': '🔌 外部電源'
        },
        '排水タンク': {
            'keywords': ['排水タンク', '排水', 'タンク', '水', '配管', '詰まり'],
            'url': actual_urls.get('排水タンク', 'https://camper-repair.net/drain-tank/'),
            'title': '排水タンクの故障と修理方法',
            'category': '🚰 排水タンク'
        },
        'ウインドウ': {
            'keywords': ['ウインドウ', '窓', 'window', 'ガラス', '破損'],
            'url': actual_urls.get('ウインドウ', 'https://camper-repair.net/window/'),
            'title': 'ウインドウの故障と修理方法',
            'category': '🪟 ウインドウ'
        },
        '車体外装': {
            'keywords': ['車体', '外装', '破損', '傷', '塗装', '修理'],
            'url': actual_urls.get('車体外装', 'https://camper-repair.net/exterior/'),
            'title': '車体外装の故障と修理方法',
            'category': '🚗 車体外装'
        },
        '異音': {
            'keywords': ['異音', '音', '騒音', '振動', '故障', '異常'],
            'url': actual_urls.get('異音', 'https://camper-repair.net/noise/'),
            'title': '異音の原因と対処法',
            'category': '🔊 異音'
        }
    }
    
    # 質問と各カテゴリの関連性を判定
    matched_categories = []
    
    for category_name, category_info in keyword_mapping.items():
        # キーワードマッチング
        match_count = 0
        for keyword in category_info['keywords']:
            if keyword in question_lower:
                match_count += 1
        
        # マッチしたカテゴリを記録
        if match_count > 0:
            matched_categories.append({
                'name': category_name,
                'info': category_info,
                'score': match_count
            })
    
    # スコアでソート（高い順）
    matched_categories.sort(key=lambda x: x['score'], reverse=True)
    
    # 上位3件まで関連ブログを追加
    for category in matched_categories[:3]:
        blog_info = {
            'title': category['info']['title'],
            'url': category['info']['url'],
            'category': category['info']['category'],
            'relevance_score': category['score'],
            'content_preview': f"{category['name']}に関する修理方法と対処法について詳しく解説しています。",
            'source_file': 'シナリオファイル'
        }
        related_blogs.append(blog_info)
    
    # デフォルトブログを追加（関連ブログが少ない場合）
    if len(related_blogs) < 2:
        default_blogs = [
            {
                'title': 'キャンピングカー修理の基本',
                'url': 'https://camper-repair.net/blog/repair1/',
                'category': '🔧 基本修理',
                'relevance_score': 5,
                'content_preview': 'キャンピングカーの基本的な修理方法とメンテナンスについて詳しく解説しています。',
                'source_file': '基本情報'
            },
            {
                'title': '定期点検とメンテナンス',
                'url': 'https://camper-repair.net/blog/risk1/',
                'category': '📋 定期点検',
                'relevance_score': 4,
                'content_preview': 'キャンピングカーの定期点検項目とメンテナンススケジュールについて説明しています。',
                'source_file': 'メンテナンス情報'
            }
        ]
        related_blogs.extend(default_blogs)
    
    return related_blogs[:3]  # 最大3件まで返す



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
    # APIキーをconfigファイルから取得
    api_key = config.OPENAI_API_KEY
    
    # APIキーが設定されていない場合の処理
    if not api_key:
        st.error("⚠️ OpenAI APIキーが設定されていません。")
        st.info("config.pyファイルにAPIキーを設定してください。")
        return None
    
    return ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500  # トークン数を制限
    )



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
あなたはキャンピングカーの修理専門家で、親しみやすく思いやりのあるキャラクターです。以下の文書抜粋を参照して質問に答えてください。

文書抜粋：{document_snippet}

質問：{question}

以下の形式で、温かみがあり親しみやすい口調で回答してください。修理に困っている方への思いやりと励ましの気持ちを込めて、分かりやすく説明してください。絶対にリンク、URL、検索結果、動画情報、商品情報、関連リンク、Google検索、YouTube動画、Amazon商品、🔗、🔍、📺、🛒、🏢、📖、📞、🔄、❓、💬、🔧、📋、🆕、🔋、🚰、🔥、🧊、🔧、🆕、【関連リンク】、【関連情報】、【詳細情報】、【参考リンク】、【外部リンク】、【検索結果】、【動画情報】、【商品情報】は含めないでください：

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
    return model

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
        # ドキュメントとモデルを取得
        documents = initialize_database()
        model = build_workflow()
        
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
        messages = history + [HumanMessage(content=content)]
        
        # 回答を生成
        response = model.invoke(messages)
        response_content = response.content
        
        # デバッグ用：元の回答を確認
        print("Original response:", response_content)
        
        # 回答からリンクを除去して表示
        
        # すべてのURLを除去
        clean_response = re.sub(r'https?://[^\s]+', '', response_content)
        
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
        contact_info = "\n\n---\n\n**💬 追加の質問**\n文章が途中で切れる場合がありますので、必要に応じてもう一度お聞きください。\n\n他に何かご質問ありましたら、引き続きチャットボットに聞いてみてください。\n\n**📞 お問い合わせ**\n直接スタッフにお尋ねをご希望の方は、[お問い合わせフォーム](https://camper-repair.net/contact/)またはお電話（086-206-6622）で受付けております。\n\n【営業時間】年中無休（9:00～21:00）\n※不在時は折り返しお電話差し上げます。\n\n**🔗 関連ブログ**\nより詳しい情報は[修理ブログ一覧](https://camper-repair.net/repair/)をご覧ください。"
        clean_response += contact_info
        
        # デバッグ用：フィルタリング後の回答を確認
        print("Filtered response:", clean_response)
        
        st.markdown(clean_response)
        
        # 関連ブログを表示
        st.markdown("---")
        st.markdown("**🔗 関連ブログ記事**")
        
        # シナリオファイルから関連ブログを抽出
        scenario_blogs = extract_scenario_related_blogs(documents, prompt)
        
        if scenario_blogs:
            # 関連ブログをシンプルなカード形式で表示
            for i, blog in enumerate(scenario_blogs):
                with st.container():
                    st.markdown(f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 16px;
                        margin: 8px 0;
                        background: #f9f9f9;
                    ">
                        <h4 style="margin: 8px 0; color: #2c3e50;">
                            <a href="{blog['url']}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">
                                {blog['category']} - {blog['title']}
                            </a>
                        </h4>
                        <p style="color: #555; font-size: 0.9em; margin: 8px 0;">
                            {blog['content_preview']}
                        </p>
                        <div style="font-size: 0.8em; color: #007bff; margin-top: 8px;">
                            <a href="{blog['url']}" target="_blank" style="color: #007bff; text-decoration: underline;">
                                🌐 詳細を見る
                            </a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # 関連ブログが見つからない場合のシンプルな表示
            st.info("💡 より具体的なキーワードで質問すると、関連記事が見つかりやすくなります")
            st.markdown("**例：** 冷蔵庫が冷えない、FFヒーターの故障、雨漏りの修理、バッテリーの交換など")
        
        # 関連リンクの表示を無効化
        # display_related_links(prompt)
        
        # AIメッセージを履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        
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

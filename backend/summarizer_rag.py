# RAGを用いて蓄積された回答の要約情報からレポートとアドバイスを生成する処理
# DBから取得した各回答の要約をDocumentに変換し、FAISSを利用してベクトル化、検索できるようにしている（ベクトルストアを作成）
# LangchainのRetrivalQAチェーンを使ってレポート＋アドバイスの生成を行う=>ベクトルストアとチェーンを組み合わせることでより関連性の高い情報を参照しながら生成する仕組み
import asyncio
from typing import Tuple
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from gpt4omini_llm import GPT4oMiniLLM

async def generate_report_with_rag(answers: list) -> Tuple[str, str]:
    """
    answers: DBから取得したAnswerオブジェクトのリスト。各オブジェクトは .summary を持つとする。
    """
    # DBから取得した各回答の要約をDocumentリストへ変換
    docs = [Document(page_content=ans.summary) for ans in answers]
    
    # 埋め込みモデルの初期化
    embeddings = OpenAIEmbeddings()  
    
    # FAISSベクトルストアの構築
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    # クエリ文を設定（プロンプト内でレポートとアドバイスを生成する指示を出す）
    query = ("以下の文書に基づいて包括的なレポートを作成します "
             "現在の関係状況について話し合い、実用的なアドバイスを提供します"
             "コミュニケーションと満足度の向上のために。回答の形式は:\n"
             "Report: <Your report text>\nAdvice: <Your advice text>")
    
    # カスタムLLM（GPT-4o-mini）を利用してRetrievalQAチェーンを作成
    llm = GPT4oMiniLLM()
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # 非同期でチェーンを実行
    result = await chain.arun(query)
    
    # 生成された結果を「Report」と「Advice」に分割（プロンプトでのフォーマットに依存）
    if "Advice:" in result:
        parts = result.split("Advice:")
        report = parts[0].replace("Report:", "").strip()
        advice = parts[1].strip()
    else:
        report = result
        advice = ""
    return report, advice

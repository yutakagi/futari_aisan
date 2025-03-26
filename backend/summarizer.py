# ユーザーから送信された回答をGPTのモデルを使用して要約するための処理を実装
import openai
from dotenv import load_dotenv # type: ignore
import os
import asyncio
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv() 
# OpenAIクライアントのインスタンスを作成
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT-4o-miniのAPI呼び出し処理を実装する
async def gpt4o_mini_call(prompt: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": "あなたは夫婦関係を専門とするコーチです."},
                {"role": "user","content":  prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Effor:{e}"

# 受け取った回答テキストに対して要約プロンプトを生成し、APIを呼び出して要約結果を返す
async def summarize_answer(answer_text: str) -> str:
    prompt = f"次の答えを要約してください: {answer_text}"
    summary = await gpt4o_mini_call(prompt)
    return summary

# 要約をもとにレポートとアドバイスを生成する
async def generate_report(combined_summary: str) -> (str, str):
    prompt = f"以下の要約に基づいて包括的なレポートと実用的なアドバイスを生成します: {combined_summary}"
    result = await gpt4o_mini_call(prompt)
    # ダミーのため、同じ結果をレポートとアドバイスに分割して返す
    report = f"Report: {result}"
    advice = f"Advice: {result}"
    return report, advice

# 非同期関数の実行例
async def main():
    answer = "夫が家事を全くやってくれず、とてもイラつきました。これから一緒に暮らしていくのはとても難しいと思いました"
    summary = await summarize_answer(answer)
    report, advice = await generate_report(summary)
    print("Summary", summary)
    print("Report:", report)
    print("Advice:",advice)

if __name__ == "__main__":
    asyncio.run(main())
// ユーザー回答を入力して送信する為のページ
// FastAPIのanswersエンドポイントにPOSTをリクエストする
"use client"

import { useState } from 'react';

export default function SubmitAnswer() {
  const [userId, setUserId] = useState("");
  const [answer, setAnswer] = useState("");
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch('http://localhost:8000/answers/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: parseInt(userId), answer_text: answer })
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    <div>
      <h1>悩みを回答</h1>
      <form onSubmit={handleSubmit}>
        <label>
          User ID:{" "}
          <input type="number" value={userId} onChange={(e) => setUserId(e.target.value)} />
        </label>
        <br />
        <label>
          悩み:{" "}
          <textarea value={answer} onChange={(e) => setAnswer(e.target.value)} />
        </label>
        <br />
        <button type="submit">回答を送る</button>
      </form>
      {response && (
        <div>
          <h2>Response</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

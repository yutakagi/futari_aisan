// ユーザーがレポートとアドバイスを取得する為のページ
// repo-to_ragエンドポイントにGETリクエストを送信し、RAG処理によって生成されたアドバイスを画面に表示
"use client"

import { useState } from 'react';

export default function Report() {
  const [userId, setUserId] = useState("");
  const [reportData, setReportData] = useState(null);

  const fetchReport = async () => {
    const res = await fetch(`http://localhost:8000/report/?user_id=${userId}`);
    const data = await res.json();
    setReportData(data);
  };

  return (
    <div>
      <h1>レポート＆アドバイス</h1>
      <label>
        User ID:{" "}
        <input type="number" value={userId} onChange={(e) => setUserId(e.target.value)} />
      </label>
      <button onClick={fetchReport}>Fetch Report</button>
      {reportData && (
        <div>
          <h2>レポート</h2>
          <p>{reportData.report}</p>
          <h2>アドバイス</h2>
          <p>{reportData.advice}</p>
        </div>
      )}
    </div>
  );
}

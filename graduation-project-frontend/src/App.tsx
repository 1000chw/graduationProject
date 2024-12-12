import React, { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import axios from "axios";

function App() {
  const [dbInfo, setDbInfo] = useState({
    host: '',
    username: '',
    password: '',
    database: '',
    table: ''
  });
  const [query, setQuery] = useState('');
  const [result, setResult] = useState<any>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setDbInfo({ ...dbInfo, [name]: value });
  };

  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuery(e.target.value);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/query', {
        dbInfo,
        query,
      });
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: 'Failed to fetch query results' });
    }
  };

  const renderTableData = (data: any[]) => {
    if (data.length === 0) return <p>No data returned</p>;

    return (
      <table className="result-table">
        <thead>
        <tr>
          <th key='idx'>index</th>
          <th key='data'>data</th>
        </tr>
        </thead>
        <tbody>
        {data.map((row, idx) => (
          <tr key={idx}>
            <td key='row-index'>{idx}</td>
            <td key='row-data'>{row}</td>
          </tr>
        ))}
        </tbody>
      </table>
    );
  };

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo"/>
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo"/>
        </a>
      </div>
      <div style={{padding: '20px'}}>
        <h1>Text-to-SQL 서비스</h1>
        <form onSubmit={handleSubmit}>
          <h2>데이터베이스 정보</h2>
          <input
            type="text"
            name="host"
            placeholder="DB 주소"
            value={dbInfo.host}
            onChange={handleInputChange}/>
          <input
            type="text"
            name="username"
            placeholder="사용자명"
            value={dbInfo.username}
            onChange={handleInputChange}/>
          <input
            type="password"
            name="password"
            placeholder="비밀번호"
            value={dbInfo.password}
            onChange={handleInputChange}/>
          <input
            type="text"
            name="database"
            placeholder="데이터베이스 이름"
            value={dbInfo.database}
            onChange={handleInputChange}/>
          <input
            type="text"
            name="table"
            placeholder="테이블 이름"
            value={dbInfo.table}
            onChange={handleInputChange}/>
          <h2>자연어 질의</h2>
          <textarea
            placeholder="질문을 입력하세요"
            value={query}
            onChange={handleQueryChange}/>
          <button type="submit">질의 실행</button>
        </form>
        {result && (
          <div className="result-container">
            <div className="result-card">
              <h3>SQL 쿼리</h3>
              <pre>{result.sql}</pre>
            </div>
            <div className="result-card">
              <h3>테이블 명</h3>
              <p>{result.table_name || "테이블 이름 없음"}</p>
            </div>
            <div className="result-card">
              <h3>실행 결과</h3>
              {renderTableData(result.data)}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default App

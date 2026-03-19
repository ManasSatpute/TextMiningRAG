import { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input) return;

    const userMsg = { role: "user", text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8000/chat", {
        question: input
      });

      const botMsg = {
        role: "bot",
        text: res.data.answer,
        sources: res.data.sources || [] // optional
      };

      setMessages(prev => [...prev, botMsg]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: "bot", text: "Error: backend not responding" }
      ]);
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>🍜 Culinary RAG Assistant</h1>

      <div style={styles.chatBox}>
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              alignSelf: m.role === "user" ? "flex-end" : "flex-start",
              background: m.role === "user" ? "#DCF8C6" : "#EEE"
            }}
          >
            <ReactMarkdown>{m.text}</ReactMarkdown>

            {/* 🔥 Show sources (RAG explainability) */}
            {m.sources && m.sources.length > 0 && (
              <div style={styles.sources}>
                <b>Sources:</b>
                <ul>
                  {m.sources.map((s, idx) => (
                    <li key={idx}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        {loading && <div>Typing...</div>}
      </div>

      <div style={styles.inputArea}>
        <input
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about your cuisine..."
        />
        <button style={styles.button} onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    fontFamily: "Arial"
  },
  header: {
    textAlign: "center",
    padding: "10px",
    background: "#222",
    color: "white"
  },
  chatBox: {
    flex: 1,
    padding: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    overflowY: "auto"
  },
  message: {
    maxWidth: "60%",
    padding: "10px",
    borderRadius: "10px"
  },
  inputArea: {
    display: "flex",
    padding: "10px",
    borderTop: "1px solid #ccc"
  },
  input: {
    flex: 1,
    padding: "10px"
  },
  button: {
    padding: "10px",
    marginLeft: "10px"
  },
  sources: {
    marginTop: "10px",
    fontSize: "12px",
    color: "#555"
  }
};

export default App;
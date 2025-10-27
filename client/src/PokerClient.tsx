import React, { useEffect, useRef, useState } from "react";

type GameState = {
  player_names?: string[];
  stacks?: number[];
  community_cards?: number[];
  current_player?: number;
  stage?: string;
  pots?: any[];
  valid_actions?: Record<string, any>;
  [key: string]: any;
};

const PokerClient: React.FC = () => {
  const [state, setState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("✅ Connected to FastAPI WebSocket!");
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const newState = JSON.parse(event.data);
      console.log("🃏 Game state update:", newState);
      setState(newState);
    };

    ws.onclose = () => {
      console.log("❌ Disconnected from server");
      setConnected(false);
    };

    return () => ws.close();
  }, []);

  const sendAction = (action: string, amount?: number) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ action, amount }));
  };

  return (
    <div style={{ padding: 20, color: "white", backgroundColor: "#111", minHeight: "100vh" }}>
      <h1 style={{ fontSize: 24, marginBottom: 10 }}>Online Poker Client</h1>
      <div style={{ marginBottom: 10 }}>
        {connected ? (
          <span style={{ color: "lightgreen" }}>Connected to server</span>
        ) : (
          <span style={{ color: "tomato" }}>Not connected</span>
        )}
      </div>

      <div>
        <button onClick={() => sendAction("fold")} style={{ margin: 5 }}>Fold</button>
        <button onClick={() => sendAction("check")} style={{ margin: 5 }}>Check</button>
        <button onClick={() => sendAction("call")} style={{ margin: 5 }}>Call</button>
        <button onClick={() => sendAction("raise", 50)} style={{ margin: 5 }}>Raise 50</button>
      </div>

      <pre style={{ background: "#222", padding: 10, marginTop: 20, borderRadius: 6 }}>
        {state ? JSON.stringify(state, null, 2) : "Waiting for game state..."}
      </pre>
    </div>
  );
};

export default PokerClient;

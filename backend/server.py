from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from engine import Game

app = FastAPI()

# CORS setup for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global poker game for demo
game = Game(player_names=["Player 1", "Player 2"], logger=None, debug=True)
clients = []

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            amount = data.get("amount")

            # Step the environment
            state = game.step(action, amount)
            
            # Broadcast state to all clients
            for client in clients:
                await client.send_json(state)

    except WebSocketDisconnect:
        clients.remove(ws)
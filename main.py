import os
from flask import Flask, request, jsonify
from logic import choose_move

app = Flask(__name__)


@app.get("/")
def on_info():
    """返回蛇的外观配置"""
    return jsonify({
        "apiversion": "1",
        "author": "dingding09",
        "color": "#1E90FF",
        "head": "default",
        "tail": "default",
        "version": "10.1.0",
    })


@app.post("/start")
def on_start():
    """游戏开始通知"""
    data = request.get_json()
    print(f"GAME START — game_id={data['game']['id']}, turn={data['turn']}")
    return jsonify({})


@app.post("/move")
def on_move():
    """每回合决策，返回移动方向"""
    data = request.get_json()
    move = choose_move(data)
    print(f"TURN {data['turn']:>4} → {move}")
    return jsonify({"move": move})


@app.post("/end")
def on_end():
    """游戏结束通知"""
    data = request.get_json()
    print(f"GAME END   — turn={data['turn']}")
    return jsonify({})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🐍 BattleSnake running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

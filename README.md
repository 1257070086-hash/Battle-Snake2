# 🐍 BattleSnake AI — Python 版

一条能跑在 [play.battlesnake.com](https://play.battlesnake.com) 上的 AI 蛇。

## 策略

1. **安全过滤**：不撞墙、不撞身体、头对头只打比自己短的蛇
2. **Flood Fill**：避免走入死路（空间不足时不进去）
3. **目标选择**：血量 < 50 时主动找食物，血量充足时抢空间

## 本地运行

```bash
cd battlesnake
pip install -r requirements.txt
python main.py
# 服务跑在 http://localhost:8000
```

## 部署到公网（Replit，免费）

1. 在 [replit.com](https://replit.com) 新建 Python Repl
2. 上传 `main.py`、`logic.py`、`requirements.txt`
3. 点击 **Run**，复制顶部的公网 URL（形如 `https://xxx.repl.co`）
4. 在 play.battlesnake.com 注册账号 → New Battlesnake → 填入 URL
5. 开始对战！

## 文件说明

| 文件 | 说明 |
|---|---|
| `main.py` | Flask HTTP 服务器，处理官方 API 的 4 个路由 |
| `logic.py` | AI 决策核心：避障 + Flood Fill + 找食物 |
| `requirements.txt` | Python 依赖（仅 Flask） |

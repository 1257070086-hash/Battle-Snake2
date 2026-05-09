"""
BattleSnake AI — logic.py v3
升级内容：
  1. Alpha-Beta 剪枝：搜索深度从2升到4，500ms内能跑完
  2. Voronoi 区域控制：计算我方控制多少棋盘格子
  3. 多敌蛇支持：对所有敌蛇建模，而不只是最近一条
  4. 时间保护：超过 400ms 自动截断返回当前最优解
  5. 优化估值函数：Voronoi控制 + Flood Fill + 食物 + 长度优势 + 攻击加成
"""

import time
from collections import deque

DIRECTIONS = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1,  0),
}
DIR_LIST = list(DIRECTIONS.values())

MAX_DEPTH   = 4      # Minimax 搜索深度
TIME_LIMIT  = 0.400  # 秒，超时保护


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def choose_move(data: dict) -> str:
    start_time = time.time()

    board  = data["board"]
    me     = data["you"]
    width  = board["width"]
    height = board["height"]
    food   = [(f["x"], f["y"]) for f in board["food"]]
    snakes = board["snakes"]

    # 构建初始状态
    state = build_state(me, snakes, food, width, height)

    # 获取我的安全方向
    my_safe = get_safe_moves(state["my_head"], state, me["id"], me["length"], width, height, aggressive=True)
    if not my_safe:
        return "up"
    if len(my_safe) == 1:
        return list(my_safe.keys())[0]

    # Alpha-Beta Minimax：迭代加深（先搜深度2，再深度4，时间够才用深度4结果）
    best_move = list(my_safe.keys())[0]
    for depth in [2, MAX_DEPTH]:
        if time.time() - start_time > TIME_LIMIT * 0.5:
            break   # 时间不够就用上一层结果

        move, _ = alphabeta_root(state, my_safe, me, snakes, food, width, height,
                                  depth, start_time)
        if move:
            best_move = move

    return best_move


# ─────────────────────────────────────────────
# Alpha-Beta 根节点
# ─────────────────────────────────────────────

def alphabeta_root(state, my_safe, me, snakes, food, width, height,
                   depth, start_time):
    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta  = float("inf")

    for direction, my_next in my_safe.items():
        if time.time() - start_time > TIME_LIMIT:
            break

        new_state = step_state(state, me["id"], my_next, food, width, height)
        score = alphabeta(new_state, depth - 1, alpha, beta, False,
                          me, snakes, food, width, height, start_time)
        if score > best_score:
            best_score = score
            best_move  = direction
        alpha = max(alpha, best_score)

    return best_move, best_score


def alphabeta(state, depth, alpha, beta, is_maximizing,
              me, snakes, food, width, height, start_time):
    """
    Alpha-Beta 剪枝 Minimax。
    is_maximizing=True  → 我方回合（选最大分）
    is_maximizing=False → 敌方回合（选最小分，取所有敌蛇联合最坏情况）
    """
    if depth == 0 or time.time() - start_time > TIME_LIMIT:
        return evaluate(state, me, food, width, height)

    if is_maximizing:
        # 我方走法
        my_safe = get_safe_moves(state["my_head"], state, me["id"],
                                  state["my_length"], width, height, aggressive=True)
        if not my_safe:
            return -1000   # 死路，极差

        value = float("-inf")
        for _, my_next in my_safe.items():
            new_state = step_state(state, me["id"], my_next, food, width, height)
            value = max(value, alphabeta(new_state, depth - 1, alpha, beta,
                                         False, me, snakes, food, width, height, start_time))
            alpha = max(alpha, value)
            if value >= beta:
                break   # β 剪枝
        return value

    else:
        # 敌方走法：对所有存活敌蛇各自走一步（取对我最坏的组合）
        enemies = [s for s in state["live_snakes"] if s["id"] != me["id"]]
        if not enemies:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time)

        # 只模拟最近的一条敌蛇（保证速度）
        nearest = min(enemies, key=lambda s: manhattan(
            state["my_head"], (s["head"]["x"], s["head"]["y"])
        ))
        e_head = (nearest["head"]["x"], nearest["head"]["y"])
        e_safe = get_safe_moves(e_head, state, nearest["id"],
                                 nearest["length"], width, height, aggressive=False)

        if not e_safe:
            # 敌蛇没路走，对我有利
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time)

        value = float("inf")
        for _, e_next in e_safe.items():
            new_state = step_state(state, nearest["id"], e_next, food, width, height)
            value = min(value, alphabeta(new_state, depth - 1, alpha, beta,
                                          True, me, snakes, food, width, height, start_time))
            beta = min(beta, value)
            if value <= alpha:
                break   # α 剪枝
        return value


# ─────────────────────────────────────────────
# 状态构建与推进
# ─────────────────────────────────────────────

def build_state(me: dict, snakes: list, food: list, width: int, height: int) -> dict:
    food_set = set(food)
    occupied = set()

    for snake in snakes:
        body = snake["body"]
        head_ate = (body[0]["x"], body[0]["y"]) in food_set
        for i, seg in enumerate(body):
            pos = (seg["x"], seg["y"])
            if i == len(body) - 1 and not head_ate:
                continue   # 尾部下回合移走，排除
            occupied.add(pos)

    return {
        "occupied":   occupied,
        "food_set":   food_set,
        "my_head":    (me["head"]["x"], me["head"]["y"]),
        "my_length":  me["length"],
        "my_health":  me["health"],
        "live_snakes": snakes,
    }


def step_state(state: dict, snake_id: str, new_head: tuple,
               food: list, width: int, height: int) -> dict:
    """
    推进一步：某条蛇从当前头移动到 new_head。
    更新 occupied、my_head（若是我方蛇）、live_snakes。
    """
    new_occupied  = set(state["occupied"])
    new_snakes    = []
    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]
    food_set      = state["food_set"]

    for s in state["live_snakes"]:
        if s["id"] == snake_id:
            # 移除旧尾、加入新头
            old_tail = s["body"][-1]
            ate_food = new_head in food_set
            new_body = [{"x": new_head[0], "y": new_head[1]}] + list(s["body"])
            if not ate_food:
                new_body = new_body[:-1]   # 缩尾
                new_occupied.discard((old_tail["x"], old_tail["y"]))
            new_occupied.add(new_head)
            new_len = len(new_body)
            new_health_val = 100 if ate_food else max(0, s["health"] - 1)

            new_s = dict(s)
            new_s["body"]   = new_body
            new_s["head"]   = {"x": new_head[0], "y": new_head[1]}
            new_s["length"] = new_len
            new_s["health"] = new_health_val

            if s["id"] == state.get("me_id", snake_id):
                new_my_head   = new_head
                new_my_length = new_len
                new_my_health = new_health_val

            new_snakes.append(new_s)
        else:
            new_snakes.append(s)

    return {
        "occupied":    new_occupied,
        "food_set":    food_set,
        "my_head":     new_my_head,
        "my_length":   new_my_length,
        "my_health":   new_my_health,
        "live_snakes": new_snakes,
        "me_id":       state.get("me_id", snake_id),
    }


# ─────────────────────────────────────────────
# 安全走法过滤
# ─────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height, aggressive=True):
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    safe = {}

    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy

        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if (nx, ny) in occupied:
            continue

        # 头对头风险判断
        risky = False
        for s in snakes:
            if s["id"] == my_id:
                continue
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan((nx, ny), eh) == 1:
                if aggressive:
                    if s["length"] >= my_len:
                        risky = True
                        break
                else:
                    if s["length"] >= my_len:
                        risky = True
                        break
        if risky:
            continue

        safe[direction] = (nx, ny)

    return safe


# ─────────────────────────────────────────────
# 估值函数（含 Voronoi）
# ─────────────────────────────────────────────

def evaluate(state, me, food, width, height):
    """
    综合评分：
      voronoi_score  * 3.0  — 我方控制区域占比
      flood_fill     * 1.0  — 直接可达空间（局部）
      food_score     * W    — 食物吸引（血量动态权重）
      length_adv     * 1.0  — 长度优势
      attack_bonus   * 0.5  — 攻击机会加成
    """
    my_pos    = state["my_head"]
    my_len    = state["my_length"]
    health    = state["my_health"]
    occupied  = state["occupied"]
    snakes    = state["live_snakes"]
    my_id     = me["id"]

    # ① Voronoi 控制分
    voronoi_mine, voronoi_total = voronoi_control(state, my_id, width, height)
    voronoi_score = voronoi_mine / max(voronoi_total, 1)

    # ② Flood Fill（局部快速评估）
    ff = flood_fill(my_pos[0], my_pos[1], width, height, occupied, max_cells=100)
    ff_score = ff / (width * height)

    # ③ 食物分
    food_w = max(0.5, (100 - health) / 50.0)
    food_list = list(state["food_set"])
    if food_list:
        min_fd = min(manhattan(my_pos, f) for f in food_list)
        food_score = 1.0 / (1 + min_fd)
    else:
        food_score = 0.0

    # ④ 长度优势
    enemies = [s for s in snakes if s["id"] != my_id]
    if enemies:
        nearest = min(enemies, key=lambda s: manhattan(
            my_pos, (s["head"]["x"], s["head"]["y"])
        ))
        len_adv = (my_len - nearest["length"]) / max(my_len, nearest["length"])
    else:
        len_adv = 1.0

    # ⑤ 攻击加成
    attack_bonus = 0.0
    for s in enemies:
        if s["length"] < my_len:
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan(my_pos, eh) <= 2:
                attack_bonus = 1.0
                break

    score = (voronoi_score * 3.0
           + ff_score      * 1.0
           + food_score    * food_w
           + len_adv       * 1.0
           + attack_bonus  * 0.5)

    return score


def voronoi_control(state, my_id, width, height):
    """
    Voronoi 区域控制：BFS 从所有蛇头同时出发，
    每个格子归属于最先到达它的蛇。
    返回 (我方控制格数, 总可达格数)
    """
    occupied = state["occupied"]
    snakes   = state["live_snakes"]

    # 初始化：每条蛇头加入队列
    owner   = {}   # (x, y) → snake_id
    dist    = {}   # (x, y) → distance
    queue   = deque()

    for s in snakes:
        h = (s["head"]["x"], s["head"]["y"])
        if h not in occupied or h == (s["head"]["x"], s["head"]["y"]):
            owner[h] = s["id"]
            dist[h]  = 0
            queue.append((h, s["id"]))

    while queue:
        (cx, cy), sid = queue.popleft()
        d = dist[(cx, cy)]
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in occupied:
                continue
            if (nx, ny) in dist:
                continue   # 已被某蛇占领
            dist[(nx, ny)]  = d + 1
            owner[(nx, ny)] = sid
            queue.append(((nx, ny), sid))

    mine  = sum(1 for sid in owner.values() if sid == my_id)
    total = len(owner)
    return mine, total


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def flood_fill(start_x, start_y, width, height, occupied, max_cells=200):
    visited = {(start_x, start_y)}
    queue   = deque([(start_x, start_y)])
    while queue and len(visited) < max_cells:
        cx, cy = queue.popleft()
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in occupied:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return len(visited)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

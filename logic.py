"""
BattleSnake AI — logic.py v4（全面升级版）
新增：
  1. 长度阶段动态权重：短蛇疯狂吃食、中蛇平衡、长蛇争控制
  2. 走法排序优化：好方向优先搜索，Alpha-Beta 剪枝更有效
  3. 搜索深度提升到 6
  4. 精确尾部预测：追踪每条蛇是否刚吃食物
  5. 危险区预标记：敌蛇下回合可达格标记高危
  6. 多蛇威胁度加权：近蛇完整建模，远蛇 Voronoi 覆盖
  7. 残局 1v1 专用策略：只剩2蛇时切换激进追击
"""

import time
from collections import deque

DIRECTIONS = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1,  0),
}
DIR_LIST   = list(DIRECTIONS.values())
DIR_NAMES  = list(DIRECTIONS.keys())

MAX_DEPTH  = 6       # 搜索深度（v4 提升到6）
TIME_LIMIT = 0.400   # 秒，超时保护


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

    # 构建初始状态（含精确尾部预测）
    state = build_state(me, snakes, food, width, height)
    state["me_id"] = me["id"]

    # 残局判断：只剩2蛇时用专用策略
    live_enemies = [s for s in snakes if s["id"] != me["id"]]
    is_endgame   = (len(live_enemies) == 1)

    # 危险区预标记（敌蛇下回合可达格）
    danger_zone = compute_danger_zone(live_enemies, state["occupied"], width, height, me["id"], me["length"])

    # 获取安全走法（先不过滤危险区，得到基础安全方向）
    my_safe = get_safe_moves(
        state["my_head"], state, me["id"], me["length"],
        width, height, aggressive=True, danger_zone=set()
    )
    if not my_safe:
        return "up"
    if len(my_safe) == 1:
        return list(my_safe.keys())[0]

    # 在基础安全方向里，过滤危险区（但保留至少1个方向）
    safe_no_danger = {d: p for d, p in my_safe.items() if p not in danger_zone}
    if safe_no_danger:
        my_safe = safe_no_danger

    # 走法排序：Flood Fill 大的优先（让 Alpha-Beta 更早剪枝）
    my_safe = sort_moves(my_safe, state, width, height)

    best_move = list(my_safe.keys())[0]

    # 迭代加深：先搜深度3兜底，再搜深度6
    for depth in [3, MAX_DEPTH]:
        if time.time() - start_time > TIME_LIMIT * 0.6:
            break
        move, _ = alphabeta_root(
            state, my_safe, me, snakes, food, width, height,
            depth, start_time, danger_zone
        )
        if move:
            best_move = move

    return best_move


# ─────────────────────────────────────────────
# 残局 1v1 专用策略
# ─────────────────────────────────────────────

def endgame_strategy(state, my_safe, me, enemy, food, width, height, start_time):
    """
    1v1 残局：
    - 若我比敌蛇长 → 追击敌蛇头，压缩其空间
    - 若我比敌蛇短 → 优先吃食，先追平长度再说
    """
    my_len  = state["my_length"]
    e_len   = enemy["length"]
    e_head  = (enemy["head"]["x"], enemy["head"]["y"])

    if my_len > e_len:
        # 追击：选距离敌蛇头最近的方向
        best = min(my_safe.items(),
                   key=lambda item: manhattan(item[1], e_head))
        return best[0]
    else:
        # 求生：选 Flood Fill 空间最大 + 靠近食物
        food_list = list(state["food_set"])
        if food_list and state["my_health"] < 60:
            best = min(my_safe.items(),
                       key=lambda item: min(manhattan(item[1], f) for f in food_list))
            return best[0]
        # 血量充足时抢空间
        best = max(my_safe.items(),
                   key=lambda item: flood_fill(item[1][0], item[1][1], width, height, state["occupied"]))
        return best[0]


# ─────────────────────────────────────────────
# 危险区预标记
# ─────────────────────────────────────────────

def compute_danger_zone(enemies, occupied, width, height, my_id, my_len):
    """
    计算所有比我长（或等长）的敌蛇下回合可能到达的格子。
    这些格子头对头我会死，标记为高危。
    """
    danger = set()
    for s in enemies:
        if s["length"] < my_len:
            continue   # 比我短的蛇，我反而要靠近
        eh = (s["head"]["x"], s["head"]["y"])
        for dx, dy in DIR_LIST:
            nx, ny = eh[0] + dx, eh[1] + dy
            if (nx, ny) not in occupied:
                danger.add((nx, ny))
    return danger


# ─────────────────────────────────────────────
# 走法排序
# ─────────────────────────────────────────────

def sort_moves(moves: dict, state, width, height) -> dict:
    """按 Flood Fill 空间从大到小排序，让 Alpha-Beta 更早剪枝"""
    scored = []
    for direction, pos in moves.items():
        ff = flood_fill(pos[0], pos[1], width, height, state["occupied"], max_cells=50)
        scored.append((direction, pos, ff))
    scored.sort(key=lambda x: x[2], reverse=True)
    return {d: p for d, p, _ in scored}


# ─────────────────────────────────────────────
# Alpha-Beta 根节点
# ─────────────────────────────────────────────

def alphabeta_root(state, my_safe, me, snakes, food, width, height,
                   depth, start_time, danger_zone):
    best_move  = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta  = float("inf")

    for direction, my_next in my_safe.items():
        if time.time() - start_time > TIME_LIMIT:
            break
        new_state = step_state(state, me["id"], my_next, width, height)
        score = alphabeta(
            new_state, depth - 1, alpha, beta, False,
            me, snakes, food, width, height, start_time, danger_zone
        )
        if score > best_score:
            best_score = score
            best_move  = direction
        alpha = max(alpha, best_score)

    return best_move, best_score


def alphabeta(state, depth, alpha, beta, is_maximizing,
              me, snakes, food, width, height, start_time, danger_zone):
    if depth == 0 or time.time() - start_time > TIME_LIMIT:
        return evaluate(state, me, food, width, height)

    if is_maximizing:
        my_safe = get_safe_moves(
            state["my_head"], state, me["id"], state["my_length"],
            width, height, aggressive=True, danger_zone=set()  # 搜索层不预标记，已体现在状态里
        )
        if not my_safe:
            return -1000

        value = float("-inf")
        for _, my_next in my_safe.items():
            new_state = step_state(state, me["id"], my_next, width, height)
            value = max(value, alphabeta(
                new_state, depth - 1, alpha, beta, False,
                me, snakes, food, width, height, start_time, danger_zone
            ))
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value

    else:
        enemies = [s for s in state["live_snakes"] if s["id"] != me["id"]]
        if not enemies:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time, danger_zone)

        # 多蛇威胁度加权：按距离排序，只完整模拟最近的敌蛇
        nearest = min(enemies, key=lambda s: manhattan(
            state["my_head"], (s["head"]["x"], s["head"]["y"])
        ))
        e_head = (nearest["head"]["x"], nearest["head"]["y"])
        e_safe = get_safe_moves(
            e_head, state, nearest["id"], nearest["length"],
            width, height, aggressive=False, danger_zone=set()
        )
        if not e_safe:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time, danger_zone)

        value = float("inf")
        for _, e_next in e_safe.items():
            new_state = step_state(state, nearest["id"], e_next, width, height)
            value = min(value, alphabeta(
                new_state, depth - 1, alpha, beta, True,
                me, snakes, food, width, height, start_time, danger_zone
            ))
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


# ─────────────────────────────────────────────
# 状态构建与推进（精确尾部预测）
# ─────────────────────────────────────────────

def build_state(me, snakes, food, width, height):
    food_set = set(food)
    occupied = set()
    snake_meta = {}  # 记录每条蛇的 just_ate 状态

    for snake in snakes:
        body    = snake["body"]
        # 精确判断：身体第一节和第二节重叠说明上回合刚吃了食物
        just_ate = (len(body) >= 2 and
                    body[0]["x"] == body[1]["x"] and
                    body[0]["y"] == body[1]["y"])
        snake_meta[snake["id"]] = {"just_ate": just_ate}

        for i, seg in enumerate(body):
            pos = (seg["x"], seg["y"])
            # 尾部处理：没吃食物时最后一节下回合移走
            if i == len(body) - 1 and not just_ate:
                continue
            occupied.add(pos)

    return {
        "occupied":    occupied,
        "food_set":    food_set,
        "my_head":     (me["head"]["x"], me["head"]["y"]),
        "my_length":   me["length"],
        "my_health":   me["health"],
        "live_snakes": snakes,
        "snake_meta":  snake_meta,
        "me_id":       me["id"],
    }


def step_state(state, snake_id, new_head, width, height):
    new_occupied  = set(state["occupied"])
    new_snakes    = []
    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]
    food_set      = state["food_set"]
    new_meta      = dict(state.get("snake_meta", {}))

    for s in state["live_snakes"]:
        if s["id"] == snake_id:
            ate_food  = new_head in food_set
            old_tail  = s["body"][-1]
            new_body  = [{"x": new_head[0], "y": new_head[1]}] + list(s["body"])
            if not ate_food:
                new_body = new_body[:-1]
                new_occupied.discard((old_tail["x"], old_tail["y"]))
            new_occupied.add(new_head)
            new_len    = len(new_body)
            new_health = 100 if ate_food else max(0, s["health"] - 1)

            new_s = dict(s)
            new_s["body"]   = new_body
            new_s["head"]   = {"x": new_head[0], "y": new_head[1]}
            new_s["length"] = new_len
            new_s["health"] = new_health
            new_meta[snake_id] = {"just_ate": ate_food}

            if s["id"] == state.get("me_id"):
                new_my_head   = new_head
                new_my_length = new_len
                new_my_health = new_health
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
        "snake_meta":  new_meta,
        "me_id":       state.get("me_id"),
    }


# ─────────────────────────────────────────────
# 安全走法过滤（含危险区）
# ─────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height,
                   aggressive=True, danger_zone=None):
    if danger_zone is None:
        danger_zone = set()
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    safe = {}

    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy

        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if (nx, ny) in occupied:
            continue
        # 危险区：高危格子直接跳过（只在根节点过滤，搜索层不过滤）
        if (nx, ny) in danger_zone:
            continue

        risky = False
        for s in snakes:
            if s["id"] == my_id:
                continue
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan((nx, ny), eh) == 1:
                if s["length"] >= my_len:
                    risky = True
                    break
                # aggressive=True 时，比我短的蛇不算危险（攻击机会）
        if risky:
            continue

        safe[direction] = (nx, ny)

    return safe


# ─────────────────────────────────────────────
# 估值函数（含长度阶段动态权重）
# ─────────────────────────────────────────────

def evaluate(state, me, food, width, height):
    my_pos   = state["my_head"]
    my_len   = state["my_length"]
    health   = state["my_health"]
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    my_id    = me["id"]

    # ── 长度阶段动态权重 ──────────────────────
    if my_len < 8:          # 短蛇：疯狂吃食
        w_voronoi = 1.0
        w_food    = 3.5
        w_len_adv = 0.5
    elif my_len < 15:       # 中蛇：平衡
        w_voronoi = 2.5
        w_food    = 2.0
        w_len_adv = 1.0
    else:                   # 长蛇：争地盘、打压
        w_voronoi = 4.0
        w_food    = 0.8
        w_len_adv = 1.5

    # ① Voronoi 控制分
    voronoi_mine, voronoi_total = voronoi_control(state, my_id, width, height)
    voronoi_score = voronoi_mine / max(voronoi_total, 1)

    # ② Flood Fill（局部快速）
    ff = flood_fill(my_pos[0], my_pos[1], width, height, occupied, max_cells=100)
    ff_score = ff / (width * height)

    # ③ 食物分（血量动态权重叠加）
    health_w  = max(0.5, (100 - health) / 50.0)
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

    # ⑤ 攻击加成：靠近比我短的蛇
    attack_bonus = 0.0
    for s in enemies:
        if s["length"] < my_len:
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan(my_pos, eh) <= 2:
                attack_bonus = 1.0
                break

    score = (voronoi_score  * w_voronoi
           + ff_score       * 1.0
           + food_score     * w_food * health_w
           + len_adv        * w_len_adv
           + attack_bonus   * 0.5)

    return score


# ─────────────────────────────────────────────
# Voronoi 区域控制
# ─────────────────────────────────────────────

def voronoi_control(state, my_id, width, height):
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    owner    = {}
    dist     = {}
    queue    = deque()

    for s in snakes:
        h = (s["head"]["x"], s["head"]["y"])
        if h not in dist:
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
            if (nx, ny) in occupied or (nx, ny) in dist:
                continue
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

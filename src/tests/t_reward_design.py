import random
import os
import matplotlib.pyplot as plt

def random_matrix(n, m):
    """生成一个n行m列的随机矩阵"""
    randmatrix = [[random.randint(0, 1) for _ in range(m)] for _ in range(n)]
    randmatrix[0][0] = 0
    return randmatrix

def complementary_matrix(matrix):
    """矩阵取补集"""
    return [[1 - x for x in row] for row in matrix]

def intersection(A, B):
    """两个矩阵的交集"""
    rows = len(A)
    cols = len(A[0])
    return [[1 if A[i][j] == 1 and B[i][j] == 1 else 0 for j in range(cols)]
            for i in range(rows)]

def instruction_fun(score):
    if score <= 0.5:
        return score
    else:
        return 1 - score

def contain_score(A, B):
    """计算两个矩阵的包含得分"""
    comple_A = complementary_matrix(A)
    inter = intersection(comple_A, B)
    count_inter = sum(sum(row) for row in inter)
    count_A = sum(sum(row) for row in A)  # A里1的个数
    count_B = sum(sum(row) for row in B)
    return 1 - (count_inter / count_B)

def score(A, B):
    """计算两个矩阵的得分"""
    return contain_score(A, B) * ratio_score(A, B)

def ratio_score(A, B):
    """计算 |A∩B| / (|A| * |B|) """
    inter = intersection(A, B)
    count_inter = sum(sum(row) for row in inter)  # 交集里1的个数
    count_A = sum(sum(row) for row in A)          # A里1的个数
    count_B = sum(sum(row) for row in B)          # B里1的个数

    if count_A == 0 or count_B == 0:
        return 0  # 避免除以0

    return instruction_fun(count_inter / count_A)


if __name__ == "__main__":
    # 固定A
    # A = [[1, 0, 0, 0],
    #      [0, 0, 1, 1],
    #      [0, 1, 1, 1],
    #      [0, 0, 0, 0]]
    A = [[0, 1, 1, 1],
         [1, 1, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 1, 1]]

    num_trials = 100000
    results = []

    for _ in range(num_trials):
        B = random_matrix(len(A), len(A[0]))
        cs = contain_score(A, B)
        rs = ratio_score(A, B)
        s = cs * rs
        results.append((s, cs, rs, B))

    # 按分数从高到低排序
    results.sort(key=lambda x: x[0], reverse=True)

    top_n = 10  # 取前10个不同分数
    top_scores = []  # 保存分数
    picked = []      # 保存模式列表

    for entry in results:
        score_val = entry[0]
        added = False
        # 判断是否已有该分数
        for idx, ts in enumerate(top_scores):
            if abs(score_val - ts) < 1e-10:
                picked[idx].append(entry)
                added = True
                break
        if not added:
            if len(top_scores) < top_n:
                top_scores.append(score_val)
                picked.append([entry])
            else:
                # 已有top10分数且没有匹配，直接跳过
                continue

    # 去重并取4个不相同矩阵
    final_list = []
    for modes in picked:
        seen_matrices = set()
        unique_modes = []
        for m in modes:
            m_key = tuple(tuple(row) for row in m[3])  # 转tuple-of-tuples做去重
            if m_key not in seen_matrices:
                seen_matrices.add(m_key)
                unique_modes.append(m)
            if len(unique_modes) >= 4:
                break
        final_list.extend(unique_modes)

    # 生成可视化
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(top_n, 4, figsize=(12, top_n * 2.2))
    fig.suptitle(f"Top {top_n} Scores - 4 Unique Patterns Each\n(1=White, 0=Black)", fontsize=16)

    for ax, (s, cs, rs, B) in zip(axes.flat, final_list):
        ax.imshow(B, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Score:{s:.4f}\nContain:{cs:.4f}  Ratio:{rs:.4f}", fontsize=8)
        ax.axis('off')

    for ax in axes.flat[len(final_list):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/top10_scores_4unique_patterns.png", dpi=200)
    plt.close()

    print(f"✅ 已保存 results/top{top_n}_scores_4unique_patterns.png")

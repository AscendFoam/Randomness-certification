import numpy as np

print("测试边界生成逻辑")
print("=" * 60)

for n_bins in [2, 3, 4]:
    print(f"\nn_bins = {n_bins}:")
    range_val = 5.0

    # 当前实现
    boundaries = np.linspace(-range_val, range_val, n_bins - 1)
    c = np.concatenate(([-np.inf], boundaries, [np.inf]))

    print(f"  boundaries 数量: {len(boundaries)} (应该是 {n_bins - 1})")
    print(f"  boundaries: {boundaries}")
    print(f"  c 数量: {len(c)} (应该是 {n_bins + 1})")
    print(f"  c: {c}")

    # 检查：n_bins个区间需要n+1个边界点
    # 区间: [c[0], c[1]), [c[1], c[2]), ..., [c[n_bins-1], c[n_bins]]
    if len(c) != n_bins + 1:
        print(f"  ❌ 错误：需要 {n_bins + 1} 个边界点，但只有 {len(c)} 个")
    else:
        print(f"  ✓ 正确：有 {n_bins + 1} 个边界点")

print("\n" + "=" * 60)
print("分析：")
print("n_bins=2 时，只有1个内部边界点（0.0）")
print("这会创建2个区间：[-inf, 0.0) 和 [0.0, +inf)")
print("但问题是：边界设置可能不合理，导致概率计算出现问题")

print("\n测试概率计算：")
from scipy.special import erf

n_bins = 2
range_val = 5.0
mu = 0.5
alpha = np.sqrt(mu)

boundaries = np.linspace(-range_val, range_val, n_bins - 1)
c = np.concatenate(([-np.inf], boundaries, [np.inf]))

print(f"\nμ = {mu}, α = {alpha:.4f}")
print(f"边界 c = {c}")

# 计算 P((0,0) | s1=+1, s2=+1)
s1, s2 = 1, 1
k, l = 0, 0

# X+ 部分
term1_upper = (c[k+1] / np.sqrt(2)) - s1 * alpha - s2 * alpha
term1_lower = (c[k] / np.sqrt(2)) - s1 * alpha - s2 * alpha
term1 = 0.5 * (erf(term1_upper) - erf(term1_lower))

print(f"\nX+ 积分区间: [{c[k]}, {c[k+1]})")
print(f"  均值偏移: {s1 * alpha + s2 * alpha:.4f}")
print(f"  term1_upper = {term1_upper}")
print(f"  term1_lower = {term1_lower}")
print(f"  erf(upper) = {erf(term1_upper):.6f}")
print(f"  erf(lower) = {erf(term1_lower):.6f}")
print(f"  term1 = {term1:.6f}")

# P- 部分
d = c.copy()
term2_upper = d[l+1] / np.sqrt(2)
term2_lower = d[l] / np.sqrt(2)
term2 = 0.5 * (erf(term2_upper) - erf(term2_lower))

print(f"\nP- 积分区间: [{d[l]}, {d[l+1]})")
print(f"  均值偏移: 0")
print(f"  term2_upper = {term2_upper}")
print(f"  term2_lower = {term2_lower}")
print(f"  erf(upper) = {erf(term2_upper):.6f}")
print(f"  erf(lower) = {erf(term2_lower):.6f}")
print(f"  term2 = {term2:.6f}")

prob = term1 * term2
print(f"\nP((0,0) | s1=+1, s2=+1) = {prob:.6f}")

if prob < 1e-6:
    print("❌ 概率接近0，这是不正常的！")
    print("\n问题：当k=0时，积分区间是 [-inf, 0.0)")
    print("      但均值在 2*α ≈ {:.4f} 处".format(2*alpha))
    print("      所以几乎所有概率都在正半轴，导致这个区间概率≈0")

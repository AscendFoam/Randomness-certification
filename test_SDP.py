#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDP 求解器测试脚本

这个脚本用于快速测试 SDP.py 的各个组件是否正常工作
"""

import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from SDP import PhysicsEngine, solve_mdi_sdp, run_single_optimization

def test_physics_engine():
    """测试物理引擎"""
    print("\n" + "="*60)
    print("测试 1: PhysicsEngine 类")
    print("="*60)

    # 使用小参数快速测试
    engine = PhysicsEngine(mu=0.5, n_bins=2, range_val=5.0)

    # 测试概率计算
    print("\n检查条件概率计算...")
    prob = engine.get_conditional_prob(s1=1, s2=1, k=0, l=0)
    print(f"  P((0,0)|s1=+1,s2=+1) = {prob:.6f}")
    assert 0 <= prob <= 1, "概率值超出[0,1]范围"
    print("  ✓ 概率值在有效范围内")

    # 测试量子态向量
    print("\n检查量子态向量...")
    v1 = engine.get_input_state_vector(s=1)
    v2 = engine.get_input_state_vector(s=-1)

    # 检查归一化
    norm1 = np.sum(np.abs(v1)**2)
    norm2 = np.sum(np.abs(v2)**2)
    print(f"  |ψ(+1)| = {norm1:.6f}")
    print(f"  |ψ(-1)| = {norm2:.6f}")
    assert abs(norm1 - 1.0) < 1e-10, "向量未归一化"
    assert abs(norm2 - 1.0) < 1e-10, "向量未归一化"
    print("  ✓ 向量正确归一化")

    # 测试密度矩阵
    print("\n检查密度矩阵...")
    rho = engine.get_joint_rho(s1=1, s2=1)

    # 检查迹
    trace = np.trace(rho)
    print(f"  Tr(ρ) = {trace:.6f}")
    assert abs(trace - 1.0) < 1e-10, "密度矩阵迹不为1"
    print("  ✓ 迹为1")

    # 检查半正定性
    eigenvals = np.linalg.eigvalsh(rho)
    min_eig = np.min(eigenvals)
    print(f"  最小特征值 = {min_eig:.10f}")
    assert min_eig >= -1e-10, "存在负特征值"
    print("  ✓ 半正定")

    # 检查厄米性
    is_hermitian = np.allclose(rho, rho.conj().T)
    assert is_hermitian, "密度矩阵不是厄米矩阵"
    print("  ✓ 厄米矩阵")

    # 测试数据生成
    print("\n检查数据生成...")
    P_obs, Rho_states, p_e = engine.generate_data(verbose=False)

    # 检查概率归一化
    print(f"  P_obs 形状: {P_obs.shape}")
    for x in range(2):
        for y in range(2):
            prob_sum = np.sum(P_obs[:, x, y])
            assert abs(prob_sum - 1.0) < 1e-4, f"P(e|x={x},y={y})求和不为1"
    print("  ✓ 所有P(e|x,y)正确归一化")

    # 检查p(e)归一化
    p_e_sum = np.sum(p_e)
    print(f"  Σ p(e) = {p_e_sum:.6f}")
    assert abs(p_e_sum - 1.0) < 1e-6, "p(e)求和不为1"
    print("  ✓ p(e)正确归一化")

    print("\n✅ PhysicsEngine 测试通过！")
    return True


def test_sdp_small():
    """测试小规模SDP求解"""
    print("\n" + "="*60)
    print("测试 2: 小规模 SDP 求解")
    print("="*60)

    print("\n运行 n_bins=2 的小规模测试...")
    print("（这应该在几秒内完成）")

    try:
        result, results_dict = run_single_optimization(
            mu_val=0.5,
            n_bins_val=2,
            range_val=5.0,
            target_idx=(0, 0),
            num_threads=4,  # 使用较少线程加快测试
            verbose=False,
            save_results=False  # 测试时不保存
        )

        print(f"\n求解状态: {results_dict['status']}")
        print(f"目标函数值: {result:.8f}")
        print(f"求解时间: {results_dict['solve_time']:.2f} 秒")

        # 检查求解状态
        assert results_dict['status'] in ['optimal', 'optimal_inaccurate'], \
            f"求解失败，状态: {results_dict['status']}"
        print("  ✓ 求解成功")

        # 检查结果范围
        assert 0 < result <= 1, "目标函数值超出合理范围"
        print("  ✓ 结果在合理范围内")

        # 计算随机性
        if result > 0:
            h_min = -np.log2(result)
            print(f"\n最小熵 H_min = {h_min:.6f} bits")
            assert h_min > 0, "最小熵应为正值"
            print("  ✓ 随机性计算正确")

        print("\n✅ 小规模 SDP 求解测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ SDP 求解失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_consistency():
    """测试结果一致性"""
    print("\n" + "="*60)
    print("测试 3: 结果一致性")
    print("="*60)

    print("\n运行相同参数两次，检查结果是否一致...")

    params = {
        'mu_val': 0.3,
        'n_bins_val': 2,
        'range_val': 5.0,
        'target_idx': (0, 0),
        'num_threads': 4,
        'verbose': False,
        'save_results': False
    }

    result1, _ = run_single_optimization(**params)
    result2, _ = run_single_optimization(**params)

    print(f"第一次运行: {result1:.10f}")
    print(f"第二次运行: {result2:.10f}")
    print(f"差异: {abs(result1 - result2):.2e}")

    assert abs(result1 - result2) < 1e-6, "相同参数的结果不一致"
    print("  ✓ 结果可重现")

    print("\n✅ 一致性测试通过！")
    return True


def test_parameter_range():
    """测试不同参数下的行为"""
    print("\n" + "="*60)
    print("测试 4: 参数范围测试")
    print("="*60)

    print("\n测试不同 μ 值...")
    mu_values = [0.1, 0.5, 1.0, 2.0]
    results = []

    for mu in mu_values:
        result, _ = run_single_optimization(
            mu_val=mu,
            n_bins_val=2,
            range_val=5.0,
            num_threads=4,
            verbose=False,
            save_results=False
        )
        results.append(result)
        print(f"  μ = {mu:.1f}: G_MDI = {result:.6f}, H_min = {-np.log2(result):.4f} bits")

    # 检查所有结果都在合理范围
    assert all(0 < r <= 1 for r in results), "存在超出范围的结果"
    print("  ✓ 所有结果在有效范围内")

    print("\n✅ 参数范围测试通过！")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("SDP 求解器完整测试套件")
    print("="*60)

    tests = [
        ("PhysicsEngine", test_physics_engine),
        ("小规模SDP求解", test_sdp_small),
        ("结果一致性", test_result_consistency),
        ("参数范围", test_parameter_range),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")

    all_passed = all(success for _, success in results)

    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！代码可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

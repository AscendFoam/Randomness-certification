#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SDP 求解器性能基准测试

测试不同参数下的求解时间和资源使用情况
"""

import numpy as np
import sys
import os
import time
import platform

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from SDP import run_single_optimization

def get_system_info():
    """获取系统信息"""
    print("=" * 60)
    print("系统信息")
    print("=" * 60)
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    print(f"Python版本: {platform.python_version()}")

    try:
        import psutil
        print(f"CPU核心数: {psutil.cpu_count(logical=False)} 物理核心")
        print(f"CPU线程数: {psutil.cpu_count(logical=True)} 逻辑处理器")
        print(f"总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        print("(安装 psutil 可查看更多系统信息: pip install psutil)")

    print()


def benchmark_scaling():
    """测试问题规模扩展性"""
    print("=" * 60)
    print("基准测试 1: 问题规模扩展性")
    print("=" * 60)
    print("\n测试不同 n_bins 下的求解时间和内存使用\n")

    test_cases = [
        {'n_bins': 2, 'desc': '小规模 (原型测试)'},
        {'n_bins': 3, 'desc': '中等规模 (推荐日常使用)'},
        {'n_bins': 4, 'desc': '大规模 (高精度)'},
    ]

    # 固定参数
    mu_val = 0.5
    range_val = 10.0
    num_threads = min(24, os.cpu_count() or 4)

    print(f"固定参数: μ = {mu_val}, range = ±{range_val}, 线程数 = {num_threads}\n")
    print(f"{'n_bins':>8} {'变量数':>10} {'约束数':>10} {'构建(s)':>12} {'求解(s)':>12} {'总时间(s)':>12} {'内存估计':>12}")
    print("-" * 88)

    results = []

    for case in test_cases:
        n_bins = case['n_bins']

        try:
            # 运行优化
            result, results_dict = run_single_optimization(
                mu_val=mu_val,
                n_bins_val=n_bins,
                range_val=range_val,
                num_threads=num_threads,
                verbose=False,
                save_results=False
            )

            # 估算内存使用
            n_vars = results_dict['num_variables']
            n_constraints = results_dict['num_constraints']
            # 粗略估算：每个4x4复数矩阵约256字节
            mem_mb = (n_vars * 16 * 16 * 8 + n_constraints * 100) / (1024**2)

            # 记录结果
            results.append({
                'n_bins': n_bins,
                'num_vars': n_vars,
                'num_constraints': n_constraints,
                'build_time': results_dict['build_time'],
                'solve_time': results_dict['solve_time'],
                'total_time': results_dict['total_time'],
                'memory_mb': mem_mb,
                'optimal_value': result,
                'status': results_dict['status']
            })

            # 打印结果
            print(f"{n_bins:>8} {n_vars:>10} {n_constraints:>10} "
                  f"{results_dict['build_time']:>12.2f} "
                  f"{results_dict['solve_time']:>12.2f} "
                  f"{results_dict['total_time']:>12.2f} "
                  f"{mem_mb:>10.0f} MB")

        except Exception as e:
            print(f"{n_bins:>8} {'失败':>10} - 错误: {str(e)[:40]}")

    print("\n" + case['desc'])

    # 分析扩展性
    if len(results) >= 2:
        print("\n扩展性分析:")
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]

            var_ratio = curr['num_vars'] / prev['num_vars']
            time_ratio = curr['solve_time'] / prev['solve_time']

            print(f"  n_bins {prev['n_bins']}→{curr['n_bins']}: "
                  f"变量数 ×{var_ratio:.1f}, 求解时间 ×{time_ratio:.1f}")

    return results


def benchmark_photon_number():
    """测试不同平均光子数的性能"""
    print("\n" + "=" * 60)
    print("基准测试 2: 平均光子数影响")
    print("=" * 60)
    print("\n测试不同 μ 值下的求解时间\n")

    mu_range = [0.1, 0.5, 1.0, 2.0, 5.0]
    n_bins = 3  # 固定中等规模
    num_threads = min(24, os.cpu_count() or 4)

    print(f"固定参数: n_bins = {n_bins}, 线程数 = {num_threads}\n")
    print(f"{'μ':>8} {'G_MDI':>15} {'H_min(bits)':>15} {'求解时间(s)':>15} {'状态':>10}")
    print("-" * 68)

    results = []

    for mu in mu_range:
        try:
            result, results_dict = run_single_optimization(
                mu_val=mu,
                n_bins_val=n_bins,
                range_val=10.0,
                num_threads=num_threads,
                verbose=False,
                save_results=False
            )

            h_min = -np.log2(result) if result > 0 else np.nan

            results.append({
                'mu': mu,
                'g_mdi': result,
                'h_min': h_min,
                'solve_time': results_dict['solve_time'],
                'status': results_dict['status']
            })

            print(f"{mu:>8.2f} {result:>15.8f} {h_min:>15.6f} "
                  f"{results_dict['solve_time']:>15.2f} {results_dict['status']:>10}")

        except Exception as e:
            print(f"{mu:>8.2f} {'失败':>15} - {str(e)[:30]}")

    # 找到最佳参数
    if results:
        best = max(results, key=lambda x: x['h_min'] if not np.isnan(x['h_min']) else 0)
        print(f"\n最佳参数: μ = {best['mu']:.2f}, H_min = {best['h_min']:.4f} bits")

    return results


def benchmark_threading():
    """测试多线程性能"""
    print("\n" + "=" * 60)
    print("基准测试 3: 多线程性能")
    print("=" * 60)
    print("\n测试不同线程数下的求解时间\n")

    max_threads = os.cpu_count() or 4
    thread_counts = [1, 2, 4, 8, max_threads] if max_threads >= 8 else [1, 2, 4, max_threads]
    thread_counts = sorted(list(set(thread_counts)))  # 去重和排序

    # 固定参数：使用中等规模问题
    mu_val = 0.5
    n_bins = 3

    print(f"固定参数: μ = {mu_val}, n_bins = {n_bins}\n")
    print(f"{'线程数':>10} {'求解时间(s)':>15} {'加速比':>12} {'效率':>12}")
    print("-" * 52)

    results = []
    baseline_time = None

    for num_threads in thread_counts:
        try:
            result, results_dict = run_single_optimization(
                mu_val=mu_val,
                n_bins_val=n_bins,
                range_val=10.0,
                num_threads=num_threads,
                verbose=False,
                save_results=False
            )

            solve_time = results_dict['solve_time']

            if baseline_time is None:
                baseline_time = solve_time
                speedup = 1.0
                efficiency = 100.0
            else:
                speedup = baseline_time / solve_time
                efficiency = (speedup / num_threads) * 100

            results.append({
                'num_threads': num_threads,
                'solve_time': solve_time,
                'speedup': speedup,
                'efficiency': efficiency
            })

            print(f"{num_threads:>10} {solve_time:>15.2f} {speedup:>12.2f}× {efficiency:>11.1f}%")

        except Exception as e:
            print(f"{num_threads:>10} {'失败':>15} - {str(e)[:30]}")

    # 推荐配置
    if results:
        # 找到效率 >= 70% 的最大线程数
        good_configs = [r for r in results if r['efficiency'] >= 70]
        if good_configs:
            recommended = max(good_configs, key=lambda x: x['num_threads'])
            print(f"\n推荐配置: {recommended['num_threads']} 线程 "
                  f"(加速比 {recommended['speedup']:.1f}×, 效率 {recommended['efficiency']:.0f}%)")

    return results


def run_all_benchmarks():
    """运行所有基准测试"""
    print("\n" + "=" * 60)
    print("SDP 求解器性能基准测试套件")
    print("=" * 60)
    print()

    # 系统信息
    get_system_info()

    # 运行基准测试
    benchmarks = [
        ("问题规模扩展性", benchmark_scaling),
        ("平均光子数影响", benchmark_photon_number),
        ("多线程性能", benchmark_threading),
    ]

    all_results = {}

    for name, benchmark_func in benchmarks:
        try:
            results = benchmark_func()
            all_results[name] = results
        except Exception as e:
            print(f"\n❌ 基准测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()

    # 生成总结报告
    print("\n" + "=" * 60)
    print("基准测试总结")
    print("=" * 60)

    if 'num_threads' in locals():
        print(f"\n推荐配置（基于当前系统）:")
        print(f"  - 日常使用: n_bins = 3, μ = 0.5-1.0")
        print(f"  - 高精度: n_bins = 4, μ = 1.0-2.0")
        print(f"  - 快速测试: n_bins = 2, μ = 0.5")

    print(f"\n系统性能:")
    max_threads = os.cpu_count() or 4
    print(f"  - CPU线程数: {max_threads}")

    if all_results.get("问题规模扩展性"):
        largest = all_results["问题规模扩展性"][-1]
        print(f"  - 最大测试规模: n_bins = {largest['n_bins']}, "
              f"求解时间 {largest['solve_time']:.1f}s")

    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    # 检查是否安装了必要的包
    try:
        import cvxpy
        import mosek
    except ImportError as e:
        print(f"错误: 缺少必要的包 - {e}")
        print("请先运行: pip install -r requirements.txt")
        sys.exit(1)

    # 运行基准测试
    results = run_all_benchmarks()

    # 可选：保存结果
    try:
        import json
        from datetime import datetime

        os.makedirs("benchmark_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results/benchmark_{timestamp}.json"

        # 转换为可JSON序列化的格式
        json_results = {}
        for key, value in results.items():
            if isinstance(value, list):
                json_results[key] = [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                     for k, v in item.items()}
                                    for item in value]

        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n基准测试结果已保存到: {filename}")

    except Exception as e:
        print(f"\n保存结果失败: {e}")

#!/usr/bin/env python
"""
验证可重复性修复的测试脚本。

使用方法：
    cd draft/lly
    python test_reproducibility.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def compute_file_hash(filepath: Path) -> str:
    """计算文件的 SHA256 哈希值。"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compare_outputs(dir1: Path, dir2: Path, patterns: list[str]) -> dict:
    """
    比较两个输出目录中的文件。

    Parameters
    ----------
    dir1 : Path
        第一次运行的输出目录
    dir2 : Path
        第二次运行的输出目录
    patterns : list[str]
        要比较的文件模式列表

    Returns
    -------
    dict
        比较结果
    """
    results = {"identical": [], "different": [], "missing": []}

    for pattern in patterns:
        files1 = sorted(dir1.glob(pattern))
        for file1 in files1:
            rel_path = file1.relative_to(dir1)
            file2 = dir2 / rel_path

            if not file2.exists():
                results["missing"].append(str(rel_path))
                continue

            hash1 = compute_file_hash(file1)
            hash2 = compute_file_hash(file2)

            if hash1 == hash2:
                results["identical"].append(str(rel_path))
            else:
                results["different"].append(str(rel_path))

    return results


def test_task1_reproducibility() -> bool:
    """测试 Task 1 的可重复性。"""
    print("\n" + "=" * 70)
    print("测试 Task 1 可重复性")
    print("=" * 70)

    # 输出目录
    outputs_dir = Path("task1_bayesian_mcmc/outputs")
    figures_dir = Path("task1_bayesian_mcmc/figures")

    # 备份第一次运行的结果
    outputs_backup = Path("task1_bayesian_mcmc/outputs_backup")
    figures_backup = Path("task1_bayesian_mcm/figures_backup")

    # 运行两次
    print("\n[1/3] 第一次运行...")
    result = subprocess.run(
        [
            sys.executable,
            "task1_bayesian_mcmc/run_inference.py",
            "--skip-sensitivity",
            "--skip-stability",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"错误：第一次运行失败\n{result.stderr}")
        return False

    # 备份结果
    import shutil

    if outputs_backup.exists():
        shutil.rmtree(outputs_backup)
    if figures_backup.exists():
        shutil.rmtree(figures_backup)

    shutil.copytree(outputs_dir, outputs_backup)
    shutil.copytree(figures_dir, figures_backup)

    print("\n[2/3] 第二次运行...")
    result = subprocess.run(
        [
            sys.executable,
            "task1_bayesian_mcmc/run_inference.py",
            "--skip-sensitivity",
            "--skip-stability",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"错误：第二次运行失败\n{result.stderr}")
        return False

    # 比较结果
    print("\n[3/3] 比较结果...")

    # 比较 JSON 文件
    json_results = compare_outputs(outputs_backup, outputs_dir, ["*.json"])

    # 比较 CSV 文件
    csv_results = compare_outputs(outputs_backup, outputs_dir, ["*.csv"])

    # 比较图片文件
    img_results = compare_outputs(figures_backup, figures_dir, ["*.png"])

    # 打印结果
    all_identical = True
    for name, results in [
        ("JSON", json_results),
        ("CSV", csv_results),
        ("图片", img_results),
    ]:
        print(f"\n{name} 文件:")
        print(f"  ✓ 完全一致: {len(results['identical'])}")
        print(f"  ✗ 存在差异: {len(results['different'])}")
        print(f"  ? 缺失文件: {len(results['missing'])}")

        if results["different"]:
            all_identical = False
            print(f"    差异文件: {', '.join(results['different'])}")

    # 清理备份
    shutil.rmtree(outputs_backup)
    shutil.rmtree(figures_backup)

    return all_identical


def test_task3_reproducibility() -> bool:
    """测试 Task 3 的可重复性。"""
    print("\n" + "=" * 70)
    print("测试 Task 3 可重复性")
    print("=" * 70)

    # 输出目录
    outputs_dir = Path("task3_mixed_effects/outputs")
    figures_dir = Path("task3_mixed_effects/figures")

    # 备份第一次运行的结果
    outputs_backup = Path("task3_mixed_effects/outputs_backup")
    figures_backup = Path("task3_mixed_effects/figures_backup")

    # 运行两次
    print("\n[1/3] 第一次运行...")
    result = subprocess.run(
        [sys.executable, "task3_mixed_effects/run_analysis.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"错误：第一次运行失败\n{result.stderr}")
        return False

    # 备份结果
    import shutil

    if outputs_backup.exists():
        shutil.rmtree(outputs_backup)
    if figures_backup.exists():
        shutil.rmtree(figures_backup)

    shutil.copytree(outputs_dir, outputs_backup)
    shutil.copytree(figures_dir, figures_backup)

    print("\n[2/3] 第二次运行...")
    result = subprocess.run(
        [sys.executable, "task3_mixed_effects/run_analysis.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"错误：第二次运行失败\n{result.stderr}")
        return False

    # 比较结果
    print("\n[3/3] 比较结果...")

    # 比较 JSON 文件
    json_results = compare_outputs(outputs_backup, outputs_dir, ["*.json"])

    # 比较 CSV 文件
    csv_results = compare_outputs(outputs_backup, outputs_dir, ["*.csv"])

    # 比较图片文件
    img_results = compare_outputs(figures_backup, figures_dir, ["*.png"])

    # 打印结果
    all_identical = True
    for name, results in [
        ("JSON", json_results),
        ("CSV", csv_results),
        ("图片", img_results),
    ]:
        print(f"\n{name} 文件:")
        print(f"  ✓ 完全一致: {len(results['identical'])}")
        print(f"  ✗ 存在差异: {len(results['different'])}")
        print(f"  ? 缺失文件: {len(results['missing'])}")

        if results["different"]:
            all_identical = False
            print(f"    差异文件: {', '.join(results['different'])}")

    # 清理备份
    shutil.rmtree(outputs_backup)
    shutil.rmtree(figures_backup)

    return all_identical


def main() -> None:
    """主函数。"""
    print("可重复性测试")
    print("注意：此测试将运行模型两次，可能需要较长时间")

    task1_ok = test_task1_reproducibility()
    task3_ok = test_task3_reproducibility()

    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"Task 1: {'✓ 通过' if task1_ok else '✗ 失败'}")
    print(f"Task 3: {'✓ 通过' if task3_ok else '✗ 失败'}")

    if task1_ok and task3_ok:
        print("\n✓ 所有测试通过！输出文件完全可重复。")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败。请检查上述差异文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()

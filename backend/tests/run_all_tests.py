import os
import sys
import subprocess
import time

sys.stdout.reconfigure(encoding='utf-8')
tests_dir = os.path.dirname(os.path.abspath(__file__))
py_exec = sys.executable

TEST_FILES = [
    "test_agent_core.py",
    "test_multimodal_agent.py",
    "test_react_loop.py",
    "test_polygon_and_scissors.py",
    "test_cerebellum_segmentation.py",
    "test_inspection_scaffolding.py",
    "test_watershed_and_smart_brush.py",
    "test_connectivity_analysis.py"
]

def run_suite():
    print("================================================================")
    print("🚀 RadPilot 全套自动化测试套件 (Test Suite Runner)")
    print("================================================================")
    
    passed = 0
    failed = 0
    start_total = time.time()

    for t_file in TEST_FILES:
        t_path = os.path.join(tests_dir, t_file)
        print(f"\n▶ 正在运行: {t_file} ...")
        t_start = time.time()
        
        proc = subprocess.run([py_exec, t_path], capture_output=True, text=True, encoding='utf-8')
        t_elapsed = round(time.time() - t_start, 2)
        
        if proc.returncode == 0:
            print(f"✅ PASSED ({t_elapsed}s)")
            passed += 1
        else:
            print(f"❌ FAILED ({t_elapsed}s)")
            print("--- 错误输出 ---")
            print(proc.stderr or proc.stdout)
            failed += 1

    total_elapsed = round(time.time() - start_total, 2)
    print("\n================================================================")
    print(f"📊 测试汇总: 总计 {len(TEST_FILES)} | 通过 {passed} | 失败 {failed} | 总耗时 {total_elapsed}s")
    print("================================================================")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()

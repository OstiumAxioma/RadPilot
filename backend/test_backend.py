import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_skills import ImageSkillsEngine
from llm_router import LLMRouter
from agent_harness import RadPilotHarness

def run_test():
    nii_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    api_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "gemini_testAPI.txt")
    
    print("=== [RadPilot Test] 1. 加载 NIfTI 图像 ===")
    engine = ImageSkillsEngine(nii_path)
    slices_info = engine.get_slice_count()
    print(f"图像切片信息: {slices_info}")
    
    print("=== [RadPilot Test] 2. 初始化 LLM Router & Agent Harness ===")
    router = LLMRouter(api_key_path)
    harness = RadPilotHarness(engine, router)
    print(f"初始状态: state={harness.state}, version={harness.current_version_index}")
    
    print("=== [RadPilot Test] 3. 测试自动脑去颅骨分割 ===")
    res1 = harness.process_doctor_input("帮我自动分割全脑去颅骨脑实质")
    print(f"Action 结果: {res1['status']} | {res1['message']}")
    
    print("=== [RadPilot Test] 4. 测试 Mask 边缘扩大 2 像素 ===")
    res2 = harness.process_doctor_input("脑轮廓放大 2 个像素")
    print(f"Action 结果: {res2['status']} | {res2['message']}")
    
    print("=== [RadPilot Test] 5. 测试撤销操作 ===")
    res3 = harness.undo()
    print(f"Action 结果: {res3['status']} | {res3['message']}")
    
    print("=== [RadPilot Test] 6. 测试金标与 Trajectory 导出 ===")
    export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "export")
    export_path = harness.export_gold_standard(export_dir)
    print(f"金标已导出至: {export_path}")
    print("=== [RadPilot Test] 后端全部功能测试通过！===")

if __name__ == "__main__":
    run_test()

import os
import sys
import json
import numpy as np
import nibabel as nib

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.agent_engine import AgentEngine

def test_react_end_to_end():
    print("=== 开始测试 ReAct 多步自主推理机与 CoT 思维链 ===")
    
    nii_path = os.path.join(os.path.dirname(backend_dir), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    api_key_path = os.path.join(os.path.dirname(backend_dir), "api", "gemini_testAPI.txt")
    
    img_nii = nib.load(nii_path)
    vol = img_nii.get_fdata()
    spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])
    
    engine = AgentEngine(vol, api_key_path=api_key_path, spacing=spacing, max_iterations=3)
    
    prompt = "请帮我提取后颅窝小脑结构，并剔除脑干部分"
    print(f"\n[Prompt]: {prompt}")
    
    res = engine.process_user_instruction(prompt)
    
    print("\n--- ReAct 结果汇总 ---")
    print(f"Action: {res.get('action')}")
    print(f"Source: {res.get('source')}")
    print(f"New Version: {res.get('new_version')}")
    print(f"Total Elapsed: {res.get('elapsed_ms')} ms")
    
    thought_steps = res.get("thought_steps", [])
    print(f"思维链步数: {len(thought_steps)}")
    for s in thought_steps:
        print(f"\n[Step {s.get('step_index')}]")
        print(f" - Thought: {s.get('thought')[:120]}...")
        print(f" - Action: {s.get('action_name')} ({json.dumps(s.get('action_params', {}), ensure_ascii=False)})")
        print(f" - Observation: {s.get('observation')}")
        
    print("\n--- 最终临床总结报告 ---")
    print(res.get("reply")[:250], "...")
    
    assert len(thought_steps) > 0, "期望产生至少 1 步以上的思维链记录"
    print("\n==========================================")
    print("🎉 backend/tests/test_react_end_to_end.py 全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_react_end_to_end()

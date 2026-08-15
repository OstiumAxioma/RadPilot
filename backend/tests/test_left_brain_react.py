import os
import sys
import json
import nibabel as nib
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core import AgentEngine

def test_left_brain():
    nii_path = os.path.join(os.path.dirname(backend_dir), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    api_key_path = os.path.join(os.path.dirname(backend_dir), "api", "gemini_testAPI.txt")

    img = nib.load(nii_path).get_fdata()
    spacing = (1.0, 1.0, 1.0)

    engine = AgentEngine(img, api_key_path=api_key_path, spacing=spacing, max_iterations=3)

    print("=== 测试自然语言指令: [请帮我分割出左半脑脑实质] ===")
    res = engine.process_user_instruction("请帮我分割出左半脑脑实质")
    print(f"Action: {res.get('action')}")
    print(f"New Version: {res.get('new_version')}")
    print(f"Total Elapsed: {res.get('elapsed_ms')} ms")
    
    thought_steps = res.get('thought_steps', [])
    print(f"思维链步数: {len(thought_steps)}")
    for s in thought_steps:
        idx = s.get('step_index')
        t_txt = s.get('thought', '')[:100]
        act = s.get('action_name')
        obs = s.get('observation')
        print(f" - [Step {idx}] Thought: {t_txt}")
        print(f"   Action: {act}")
        print(f"   Observation: {obs}")

    print("\n--- 最终临床报告 ---")
    print(res.get('reply')[:300], "...")

if __name__ == "__main__":
    test_left_brain()

import os
import sys
import numpy as np

# 强制 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_core import GLOBAL_TOOL_REGISTRY, VersionDAG, AgentEngine, ImageContext
from image_skills import ImageSkillsEngine

def test_agent_core():
    print("=== [1/4] 验证 ToolRegistry 与 Function Declarations ===")
    tools = GLOBAL_TOOL_REGISTRY.get_all_tools()
    print(f"已注册医学工具数量: {len(tools)}")
    declarations = GLOBAL_TOOL_REGISTRY.get_function_declarations()
    assert len(declarations) >= 5, "工具声明数量不足"
    for decl in declarations:
        print(f" - Tool: {decl['name']}, 参数数量: {len(decl['parameters'].get('properties', {}))}")

    print("\n=== [2/4] 验证形态学与分割算子物理度量计算 ===")
    nii_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image", "MNI152NLin6_res-1x1x1_T1w.nii")
    img_engine = ImageSkillsEngine(nii_path)
    context = ImageContext(img_engine.volume_data, spacing=(1.0, 1.0, 1.0))

    # 执行脑实质去颅骨提取
    brain_tool = GLOBAL_TOOL_REGISTRY.get_tool("extract_brain_tissue")
    res1 = brain_tool.execute(context, region="all")
    assert res1.success, f"脑实质提取失败: {res1.message}"
    print(f"脑实质提取成功: {res1.message}")
    print(f"物理度量: {res1.observation_metrics}")
    assert res1.observation_metrics["current_volume_cm3"] > 500, "脑实质体积不符合解剖常规"

    # 执行 2mm 膨胀
    context.current_mask = res1.new_mask
    dilate_tool = GLOBAL_TOOL_REGISTRY.get_tool("morphological_dilation")
    res2 = dilate_tool.execute(context, radius_mm=2.0)
    assert res2.success, f"膨胀失败: {res2.message}"
    print(f"形态学膨胀成功: {res2.message}")
    print(f"体积增加量: {res2.observation_metrics['volume_change_mm3']} mm³")
    assert res2.observation_metrics["volume_change_mm3"] > 0

    print("\n=== [3/4] 验证 VersionDAG 有向无环图分叉与 Diff 计算 ===")
    dag = VersionDAG(img_engine.shape)
    v1 = dag.commit("BRAIN_EXTRACT", "提取脑实质", res1.new_mask, res1.observation_metrics, "extract_brain_tissue")
    v2 = dag.commit("DILATION", "外扩2毫米", res2.new_mask, res2.observation_metrics, "morphological_dilation")
    assert dag.current_node_id == "v2"

    # 撤销到 v1
    v_undo = dag.undo()
    assert v_undo.node_id == "v1"
    print(f"成功撤销到: {v_undo.node_id}")

    # 在 v1 上分叉新分支 (执行腐蚀)
    context.current_mask = v1.mask_data
    erode_tool = GLOBAL_TOOL_REGISTRY.get_tool("morphological_erosion")
    res3 = erode_tool.execute(context, radius_mm=1.5)
    v3 = dag.commit("EROSION", "收缩1.5毫米", res3.new_mask, res3.observation_metrics, "morphological_erosion")
    print(f"成功在历史节点上分叉新分支节点: {v3.node_id}, 分支名: {v3.branch_name}")
    assert v3.parent_id == "v1"

    # 计算 v2 与 v3 之间的形态学差异
    diff = dag.compute_diff("v2", "v3", context.voxel_volume_mm3)
    print(f"v2 与 v3 分支形态学差异比对: Dice={diff['dice_similarity']}, 体积差={diff['volume_difference_mm3']} mm³")

    print("\n=== [4/4] 验证 AgentEngine 与真实 Gemini Function Calling 闭环 ===")
    api_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "gemini_testAPI.txt")
    agent = AgentEngine(img_engine.volume_data, api_key_path, spacing=(1.0, 1.0, 1.0))

    # 测试自然语言指令驱动真实推理
    test_instruction = "请帮我提取全脑脑实质"
    print(f"测试输入指令: '{test_instruction}'")
    ai_result = agent.process_user_instruction(test_instruction)
    print("Agent 执行响应结果:")
    print(f" - Action: {ai_result.get('action')}")
    print(f" - Source: {ai_result.get('source')}")
    print(f" - New Version: {ai_result.get('new_version')}")
    print(f" - Elapsed: {ai_result.get('elapsed_ms')} ms")
    print(f" - Reply:\n{ai_result.get('reply')}")

    assert ai_result.get("source") in ["GEMINI_FUNCTION_CALLING", "GEMINI_REASONING"], "未走真实 Gemini 路由"
    print("\n🎉 全部 4 项核心测试均圆满通过！无任何正则与硬编码！")

if __name__ == "__main__":
    test_agent_core()

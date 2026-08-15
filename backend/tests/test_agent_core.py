import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
# 将 backend 根目录加入 Python 搜索路径
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.registry import GLOBAL_TOOL_REGISTRY
from agent_core.tools.base_tool import ImageContext
from agent_core.version_dag import VersionDAG
from agent_core.agent_engine import AgentEngine

def test_tool_registry():
    print("=== [Test 1] 测试 ToolRegistry 强类型工具注册与 Schema 导出 ===")
    tools = GLOBAL_TOOL_REGISTRY.get_all_tools()
    print(f"当前已注册工具数量: {len(tools)}")
    assert len(tools) >= 7, "已注册工具数不足"
    
    declarations = GLOBAL_TOOL_REGISTRY.get_function_declarations()
    print("成功导出 Gemini Function Calling 声明规范:")
    for d in declarations:
        print(f" - [{d['name']}]: {d['description'][:40]}...")
    assert len(declarations) == len(tools)
    print("✓ ToolRegistry 测试通过！\n")

def test_morphology_and_metrics():
    print("=== [Test 2] 测试形态学膨胀算子与物理度量计算 ===")
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)
    fake_img = np.random.uniform(100, 2000, size=shape).astype(np.float32)
    fake_mask = np.zeros(shape, dtype=np.uint8)
    fake_mask[40:60, 40:60, 40:60] = 1  # 20x20x20 = 8000 体素 (8 cm³)
    
    context = ImageContext(fake_img, fake_mask, spacing)
    dilation_tool = GLOBAL_TOOL_REGISTRY.get_tool("morphological_dilation")
    assert dilation_tool is not None, "未找到 morphological_dilation 算子"
    
    result = dilation_tool.execute(context, radius_mm=2.0)
    print(f"膨胀执行结果: 成功={result.success}, 动作={result.action_description}")
    print("物理观察指标 (Observation):")
    print(json.dumps(result.observation_metrics, indent=2, ensure_ascii=False))
    
    assert result.success is True
    assert result.observation_metrics["volume_change_mm3"] > 0
    assert result.observation_metrics["current_volume_cm3"] > 8.0
    print("✓ 形态学算子与物理度量测试通过！\n")

def test_version_dag():
    print("=== [Test 3] 测试真有向无环图 (VersionDAG) 分叉与差异计算 ===")
    shape = (100, 100, 100)
    dag = VersionDAG(shape)
    
    # 提交 v1
    mask_v1 = np.zeros(shape, dtype=np.uint8)
    mask_v1[30:50, 30:50, 30:50] = 1
    node_v1 = dag.commit(
        action_name="EXTRACT_BRAIN",
        prompt="提取脑实质",
        new_mask=mask_v1,
        metrics={"current_volume_cm3": 8.0, "voxel_delta": 8000}
    )
    print(f"提交节点 1: {node_v1.node_id} ({node_v1.action_name})")
    
    # 提交 v2
    mask_v2 = np.zeros(shape, dtype=np.uint8)
    mask_v2[30:55, 30:55, 30:55] = 1
    node_v2 = dag.commit(
        action_name="DILATION",
        prompt="外扩2毫米",
        new_mask=mask_v2,
        metrics={"current_volume_cm3": 15.625, "voxel_delta": 7625}
    )
    print(f"提交节点 2: {node_v2.node_id} ({node_v2.action_name})")
    
    # 回退到 v1 并分叉出新分支 branch_v3
    dag.checkout("v1")
    dag.create_branch("branch_v3")
    mask_v3 = np.zeros(shape, dtype=np.uint8)
    mask_v3[30:45, 30:45, 30:45] = 1
    node_v3 = dag.commit(
        action_name="EROSION",
        prompt="收缩2毫米",
        new_mask=mask_v3,
        metrics={"current_volume_cm3": 3.375, "voxel_delta": -4625}
    )
    print(f"在分叉分支提交节点 3: {node_v3.node_id} (分支={node_v3.branch_name})")
    
    # 计算 v2 与 v3 的差异
    diff = dag.compute_diff("v2", "v3")
    print("v2 与 v3 的形态学差异分析:")
    print(json.dumps(diff, indent=2, ensure_ascii=False))
    
    assert diff["dice_score"] > 0
    print("✓ VersionDAG 分叉与差异计算测试通过！\n")

if __name__ == "__main__":
    test_tool_registry()
    test_morphology_and_metrics()
    test_version_dag()
    print("==========================================")
    print("🎉 backend/tests/test_agent_core.py 全部通过！")
    print("==========================================")

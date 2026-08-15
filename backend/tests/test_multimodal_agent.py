import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from agent_core.tools.registry import GLOBAL_TOOL_REGISTRY
from agent_core.multimodal.slice_encoder import MultiModalSliceEncoder
from agent_core.tools.base_tool import ImageContext

def test_multimodal_pipeline():
    print("=== 开始测试原生多模态视觉空间 Agent ===")
    
    shape = (182, 218, 182)
    fake_img = np.random.uniform(0, 5000, size=shape).astype(np.float32)
    fake_img[40:140, 20:100, 20:80] += 3000.0
    
    print("\n[Step 1] 测试全脑 6 格多断层画廊与坐标标尺编码...")
    slices = MultiModalSliceEncoder.encode_multiview_slices(fake_img)
    assert len(slices) == 3, f"期望 3 个画廊部件，实际获得: {len(slices)}"
    for idx, s in enumerate(slices):
        data_len = len(s["inlineData"]["data"])
        print(f" - 画廊 {idx+1} 成功生成 JPEG Base64 (长度: {data_len} 字节)")
        assert data_len > 1000, "画廊图像数据过短"

    print("\n[Step 2] 测试 SpatialPromptGuidedSegmentationTool 3D 区域生长连续轮廓精修...")
    tool = GLOBAL_TOOL_REGISTRY.get_tool("spatial_prompt_guided_segmentation")
    assert tool is not None, "未找到 spatial_prompt_guided_segmentation 算子"
    
    context = ImageContext(fake_img, np.zeros(shape, dtype=np.uint8), (1.0, 1.0, 1.0))
    res = tool.execute(
        context,
        target_name="小脑",
        bbox_3d=[30, 10, 15, 150, 110, 85],
        center_point_3d=[90, 60, 50],
        tissue_intensity_type="brain_parenchyma"
    )
    print(f" - 算子执行结果: 成功={res.success}, 动作={res.action_description}")
    print(f" - 物理观测度量: 变化体积={res.observation_metrics.get('volume_change_mm3')} mm³, 标定体积={res.observation_metrics.get('current_volume_cm3')} cm³")
    assert res.success, "算子执行失败"
    assert res.observation_metrics.get("volume_change_mm3", 0) > 0, "掩码体积应当大于0"

    print("\n==========================================")
    print("🎉 backend/tests/test_multimodal_agent.py 全部通过！")
    print("==========================================")

if __name__ == "__main__":
    test_multimodal_pipeline()

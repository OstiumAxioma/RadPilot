import cv2
import base64
import numpy as np
from typing import Dict, List, Any, Optional

class MultiModalSliceEncoder:
    """
    医学三维断层视觉切片编码器
    将 3D NumPy 体数据动态切片并编码为供 Gemini 视觉接口消费的标准多模态图像 Parts
    """
import cv2
import base64
import numpy as np
from typing import Dict, List, Any, Optional

class MultiModalSliceEncoder:
    """
    医学三维断层视觉切片与全脑画廊 (Contact Sheet) 编码器
    将 3D MRI 体数据多断层阵列与解剖标尺编码为供 Gemini 视觉接口消费的高信息密度图像部件
    """

    @staticmethod
    def _normalize_and_overlay(
        slice_2d: np.ndarray,
        mask_2d: Optional[np.ndarray],
        min_val: float,
        rng: float
    ) -> np.ndarray:
        clipped = np.clip(slice_2d, min_val, min_val + rng)
        norm = ((clipped - min_val) / rng * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        
        if mask_2d is not None and np.any(mask_2d > 0):
            overlay = rgb.copy()
            overlay[mask_2d > 0] = [212, 182, 6]  # 青色高亮
            rgb = cv2.addWeighted(overlay, 0.45, rgb, 0.55, 0)
        return rgb

    @classmethod
    def encode_multiview_slices(
        cls,
        volume_data: np.ndarray,
        current_mask: Optional[np.ndarray] = None,
        ww: float = 7000.0,
        wl: float = 3500.0
    ) -> List[Dict[str, Any]]:
        """
        生成全脑高信息密度的 3 张复合解剖画廊 (Axial 6格全脑断层、Coronal 3格、Sagittal 3格)
        """
        dim_x, dim_y, dim_z = volume_data.shape
        min_val = wl - (ww / 2.0)
        max_val = wl + (ww / 2.0)
        rng = max(1.0, max_val - min_val)

        # -------------------------------------------------------------
        # 1. 轴位 6 格断层画廊 (覆盖从颅底/小脑到大脑顶叶: Z=18%, 30%, 42%, 55%, 68%, 82%)
        # -------------------------------------------------------------
        z_ratios = [0.18, 0.30, 0.42, 0.55, 0.68, 0.82]
        z_labels = [
            "Z=18% (颅底/后颅窝小脑)",
            "Z=30% (第四脑室/脑干/小脑)",
            "Z=42% (基底节/中脑/颞叶)",
            "Z=55% (侧脑室/大脑皮质)",
            "Z=68% (半卵圆中心/顶叶)",
            "Z=82% (大脑皮层穹窿顶)"
        ]
        
        axial_tiles = []
        for r, label in zip(z_ratios, z_labels):
            idx = int(np.clip(dim_z * r, 0, dim_z - 1))
            img_slice = np.rot90(volume_data[:, :, idx], 1)
            m_slice = np.rot90(current_mask[:, :, idx], 1) if current_mask is not None else None
            tile = cls._normalize_and_overlay(img_slice, m_slice, min_val, rng)
            
            # 在切片左上角叠加解剖层标尺
            cv2.putText(tile, f"Z={idx} {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            axial_tiles.append(tile)

        # 拼接成 2 行 3 列的接触网格图
        row1 = np.hstack([axial_tiles[0], axial_tiles[1], axial_tiles[2]])
        row2 = np.hstack([axial_tiles[3], axial_tiles[4], axial_tiles[5]])
        axial_contact_sheet = np.vstack([row1, row2])

        # -------------------------------------------------------------
        # 2. 冠状位 3 格画廊 (前额叶 Y=30%, 中央基底节 Y=50%, 后颅窝小脑/枕叶 Y=70%)
        # -------------------------------------------------------------
        y_ratios = [0.30, 0.50, 0.70]
        y_labels = ["Y=30% (前额叶/额骨)", "Y=50% (中线脑干/侧脑室)", "Y=70% (后枕叶/后颅窝小脑)"]
        coronal_tiles = []
        for r, label in zip(y_ratios, y_labels):
            idx = int(np.clip(dim_y * r, 0, dim_y - 1))
            img_slice = np.rot90(volume_data[:, idx, :], 1)
            m_slice = np.rot90(current_mask[:, idx, :], 1) if current_mask is not None else None
            tile = cls._normalize_and_overlay(img_slice, m_slice, min_val, rng)
            cv2.putText(tile, f"Y={idx} {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            coronal_tiles.append(tile)
        coronal_sheet = np.hstack(coronal_tiles)

        # -------------------------------------------------------------
        # 3. 矢状位 3 格画廊 (右脑 X=30%, 正中矢状面 X=50%, 左脑 X=70%)
        # -------------------------------------------------------------
        x_ratios = [0.30, 0.50, 0.70]
        x_labels = ["X=30% (右半球/小脑半球)", "X=50% (正中矢状面/脑干/小脑蚓部)", "X=70% (左半球/小脑半球)"]
        sagittal_tiles = []
        for r, label in zip(x_ratios, x_labels):
            idx = int(np.clip(dim_x * r, 0, dim_x - 1))
            img_slice = np.rot90(volume_data[idx, :, :], 1)
            m_slice = np.rot90(current_mask[idx, :, :], 1) if current_mask is not None else None
            tile = cls._normalize_and_overlay(img_slice, m_slice, min_val, rng)
            cv2.putText(tile, f"X={idx} {label}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            sagittal_tiles.append(tile)
        sagittal_sheet = np.hstack(sagittal_tiles)

        def img_to_b64(img_mat: np.ndarray) -> str:
            _, buffer = cv2.imencode('.jpg', img_mat, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')

        return [
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_to_b64(axial_contact_sheet)
                }
            },
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_to_b64(coronal_sheet)
                }
            },
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_to_b64(sagittal_sheet)
                }
            }
        ]

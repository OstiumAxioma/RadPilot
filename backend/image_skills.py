import os
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import io
import base64

class ImageSkillsEngine:
    def __init__(self, nii_path: str):
        self.nii_path = nii_path
        self.img_nii = None
        self.volume_data = None
        self.affine = None
        self.header = None
        self.shape = (0, 0, 0)
        self.current_mask_3d = None
        
        self.load_nifti(nii_path)

    def load_nifti(self, path: str):
        """加载 NIfTI 医疗图像"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"NIfTI 文件未找到: {path}")
        
        self.img_nii = nib.load(path)
        # 获取 3D float/int 数组
        data = self.img_nii.get_fdata()
        self.volume_data = np.nan_to_num(data)
        self.affine = self.img_nii.affine
        self.header = self.img_nii.header
        self.shape = self.volume_data.shape
        
        # 初始化空白 3D Mask (uint8: 0 为背景, 255 为前景)
        self.current_mask_3d = np.zeros(self.shape, dtype=np.uint8)

    def get_slice_count(self) -> dict:
        """获取图像维度与切片数量信息 (Axial, Coronal, Sagittal)"""
        return {
            "dim_x": int(self.shape[0]),
            "dim_y": int(self.shape[1]),
            "dim_z": int(self.shape[2]),
            "axial_slices": int(self.shape[2]),
            "coronal_slices": int(self.shape[1]),
            "sagittal_slices": int(self.shape[0]),
        }

    def _normalize_slice(self, slice_2d: np.ndarray, ww: float = 7000.0, wl: float = 3500.0) -> np.ndarray:
        """根据 MNI152 实际灰度标量量程 (0-9999) 计算 DICOM/NIfTI 窗宽窗位线性截断"""
        min_val = wl - (ww / 2.0)
        max_val = wl + (ww / 2.0)
        if max_val - min_val <= 1e-5:
            return np.zeros(slice_2d.shape, dtype=np.uint8)
        clipped = np.clip(slice_2d, min_val, max_val)
        norm = (clipped - min_val) / (max_val - min_val) * 255.0
        return norm.astype(np.uint8)

    def get_slice_2d(self, slice_index: int, axis: str = "axial") -> np.ndarray:
        """从 3D MRI 体数据中切出二维 Slice"""
        if axis == "axial":
            idx = max(0, min(slice_index, self.shape[2] - 1))
            return self.volume_data[:, :, idx]
        elif axis == "coronal":
            idx = max(0, min(slice_index, self.shape[1] - 1))
            return self.volume_data[:, idx, :]
        else:  # sagittal
            idx = max(0, min(slice_index, self.shape[0] - 1))
            return self.volume_data[idx, :, :]

    def get_slice_base64(self, slice_index: int, axis: str = "axial", ww: float = 7000.0, wl: float = 3500.0) -> str:
        """获取指定切片的灰度图 Base64 (支持对应 MNI152 标量的 WW/WL 窗宽窗位)"""
        slice_2d = self.get_slice_2d(slice_index, axis)
        slice_uint8 = self._normalize_slice(slice_2d, ww=ww, wl=wl)
        slice_uint8 = np.rot90(slice_uint8)
        if axis == "sagittal":
            slice_uint8 = np.fliplr(slice_uint8)  # 鼻子朝左 (Anterior→左)
        
        img = Image.fromarray(slice_uint8)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def get_mask_slice_2d(self, slice_index: int, axis: str = "axial") -> np.ndarray:
        """提取 2D Mask 切片 (已旋转对齐)"""
        if axis == "axial":
            idx = max(0, min(slice_index, self.shape[2] - 1))
            mask_slice = self.current_mask_3d[:, :, idx]
        elif axis == "coronal":
            idx = max(0, min(slice_index, self.shape[1] - 1))
            mask_slice = self.current_mask_3d[:, idx, :]
        else:
            idx = max(0, min(slice_index, self.shape[0] - 1))
            mask_slice = self.current_mask_3d[idx, :, :]

        result = np.rot90(mask_slice)
        if axis == "sagittal":
            result = np.fliplr(result)  # 与灰度图保持一致
        return result

    def get_mask_slice_base64(self, slice_index: int, axis: str = "axial") -> str:
        """获取 Mask 切片 Base64 编码"""
        mask_uint8 = self.get_mask_slice_2d(slice_index, axis)
        img = Image.fromarray(mask_uint8)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def get_triplanar_bundle(self, axial_idx: int, coronal_idx: int, sagittal_idx: int, ww: float = 7000.0, wl: float = 3500.0) -> dict:
        """打包一次性返回三视图 (Axial, Coronal, Sagittal) 的图像与 Mask Base64 (带精准 WW/WL)"""
        return {
            "axial": {
                "index": axial_idx,
                "image_base64": self.get_slice_base64(axial_idx, "axial", ww=ww, wl=wl),
                "mask_base64": self.get_mask_slice_base64(axial_idx, "axial")
            },
            "coronal": {
                "index": coronal_idx,
                "image_base64": self.get_slice_base64(coronal_idx, "coronal", ww=ww, wl=wl),
                "mask_base64": self.get_mask_slice_base64(coronal_idx, "coronal")
            },
            "sagittal": {
                "index": sagittal_idx,
                "image_base64": self.get_slice_base64(sagittal_idx, "sagittal", ww=ww, wl=wl),
                "mask_base64": self.get_mask_slice_base64(sagittal_idx, "sagittal")
            }
        }

    def get_volume_raw_base64(self) -> dict:
        """
        导出 3D Uint8 标量体素数组 (按 P99 归一化, 降采样至 96^3)

        内存布局 (dz, dy, dx) C-order -- 指导 Three.js DataTexture 读取:
          zi-chunk = 一张 Axial 切片, 内部布局 row=j(Coronal) x col=i(Sagittal)
          DataTexture(width=dx, height=dy):
            U 轴 = i = Sagittal (L-R) → 映射到水平面 Three.js X 轴
            V 轴 = j = Coronal  (P-A) → 映射到水平面 Three.js Z 轴
          dz 方向: z=0 (Inferior/脑底) → z=95 (Superior/头顶)
        """
        p99 = float(np.percentile(self.volume_data, 99))
        norm_vol = np.clip(self.volume_data, 0, p99) / p99 * 255.0
        vol = norm_vol.astype(np.uint8)

        d_x, d_y, d_z = 96, 96, 96
        # resized_vol shape: (d_x, d_y, d_z), d_x=Sagittal, d_y=Coronal, d_z=Axial
        resized_vol = np.zeros((d_x, d_y, d_z), dtype=np.uint8)
        for zi in range(d_z):
            orig_z = int(zi * (self.shape[2] - 1) / (d_z - 1))
            slice_z = vol[:, :, orig_z]            # shape (182, 218)
            resized = cv2.resize(slice_z, (d_y, d_x), interpolation=cv2.INTER_AREA)
            resized_vol[:, :, zi] = resized

        # transpose(2,1,0): (d_x,d_y,d_z) -> (d_z,d_y,d_x)
        # flip axis=1 (Coronal/j): 使 Anterior(鼻子) 在 3D 水平切面正确朝左
        vol_export = np.ascontiguousarray(np.flip(resized_vol.transpose(2, 1, 0), axis=1))  # (dz, dy, dx)
        buffer_bytes = vol_export.tobytes()
        encoded = base64.b64encode(buffer_bytes).decode("utf-8")

        return {
            "dimensions": [d_z, d_y, d_x],   # [dz=96, dy=96, dx=96]
            "spacing": [1.0, 1.0, 1.0],
            "raw_base64": encoded
        }

    def get_volume_vtk_payload(self) -> dict:
        """
        导出专供 @kitware/vtk.js vtkImageData 构造的标准体数据载荷
        采用 Fortran-order (X 变化最快) 排布，与 vtkImageData.getPointData() 原生对齐
        """
        p99 = float(np.percentile(self.volume_data, 99))
        norm_vol = np.clip(self.volume_data, 0, p99) / p99 * 255.0
        vol_uint8 = norm_vol.astype(np.uint8)

        # Fortran-order 展平，完全契合 VTK [x, y, z] 索引: idx = x + y*dim_x + z*dim_x*dim_y
        flat_bytes = np.ascontiguousarray(vol_uint8.ravel(order='F')).tobytes()
        encoded = base64.b64encode(flat_bytes).decode("utf-8")

        return {
            "dimensions": [int(self.shape[0]), int(self.shape[1]), int(self.shape[2])],
            "spacing": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "scalar_range": [0, 255],
            "raw_base64": encoded
        }

    def get_mask_vtk_payload(self) -> dict:
        """
        导出专供 @kitware/vtk.js vtkImageData 构造的标准 3D Mask 体数据载荷
        与 get_volume_vtk_payload 完全相同的几何空间与 Fortran-order 排布
        """
        mask_uint8 = self.current_mask_3d.astype(np.uint8)
        flat_bytes = np.ascontiguousarray(mask_uint8.ravel(order='F')).tobytes()
        encoded = base64.b64encode(flat_bytes).decode("utf-8")

        return {
            "dimensions": [int(self.shape[0]), int(self.shape[1]), int(self.shape[2])],
            "spacing": [1.0, 1.0, 1.0],
            "origin": [0.0, 0.0, 0.0],
            "scalar_range": [0, 255],
            "raw_base64": encoded,
            "has_mask": bool(np.any(mask_uint8 > 0))
        }

    # -------------------------------------------------------------
    # 医疗图像处理算子 (Image Tool Skills)
    # -------------------------------------------------------------
    
    def skull_strip_brain_extraction(self, region: str = "full") -> np.ndarray:
        """
        全脑 / 左脑 / 右脑 脑实质提取算子 (Brain Extraction)
        region: "full" | "left" | "right"
        """
        print(f"[Skill] 开始脑实质提取算子 (目标区域: {region})...")
        
        # 1. 全体 3D 归一化
        vol = self._normalize_slice(self.volume_data)
        
        # 2. 生成 3D 粗选 Mask (Otsu 阈值)
        mask_3d = np.zeros(self.shape, dtype=np.uint8)
        for z in range(self.shape[2]):
            slice_z = vol[:, :, z]
            if np.max(slice_z) > 10:
                _, thresh = cv2.threshold(slice_z, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
                if num_labels > 1:
                    for i in range(1, num_labels):
                        if stats[i, cv2.CC_STAT_AREA] > 100:
                            mask_3d[:, :, z][labels == i] = 255

        # 3. 3D 形态学闭运算与连通域提纯
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for z in range(self.shape[2]):
            if np.any(mask_3d[:, :, z]):
                mask_3d[:, :, z] = cv2.morphologyEx(mask_3d[:, :, z], cv2.MORPH_CLOSE, kernel)
                eroded = cv2.erode(mask_3d[:, :, z], kernel, iterations=2)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
                if num_labels > 1:
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    brain_core = np.zeros_like(eroded)
                    brain_core[labels == largest_label] = 255
                    dilated_brain = cv2.dilate(brain_core, kernel, iterations=2)
                    mask_3d[:, :, z] = dilated_brain
                else:
                    mask_3d[:, :, z] = 0

        # 4. 根据左脑 / 右脑物理空间中线 (X 轴中心) 进行精准几何裁切
        if region in ["left", "right"]:
            center_x = self.shape[0] // 2
            # 计算非零 Mask 的物理中心线进行微调
            non_zero_x = np.where(mask_3d > 0)[0]
            if len(non_zero_x) > 0:
                center_x = int(np.mean(non_zero_x))
                
            clipped_mask = np.zeros_like(mask_3d)
            if region == "left":
                # 左脑 (放射科视角: 图像左侧 X > center_x 或 X < center_x)
                clipped_mask[center_x:, :, :] = mask_3d[center_x:, :, :]
                print(f"[Skill] 沿 3D 纵裂中线 X={center_x} 空间裁切左半脑实质。")
            else:
                # 右脑
                clipped_mask[:center_x, :, :] = mask_3d[:center_x, :, :]
                print(f"[Skill] 沿 3D 纵裂中线 X={center_x} 空间裁切右半脑实质。")
            mask_3d = clipped_mask

        self.current_mask_3d = mask_3d
        print(f"[Skill] {region} 脑实质分割计算完毕。")
        return self.current_mask_3d

    def expand_mask(self, pixels: int = 2) -> np.ndarray:
        """Mask 形态学膨胀 (外扩)"""
        pixels = max(1, min(pixels, 20))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
        
        for z in range(self.shape[2]):
            if np.any(self.current_mask_3d[:, :, z]):
                self.current_mask_3d[:, :, z] = cv2.dilate(self.current_mask_3d[:, :, z], kernel)
        return self.current_mask_3d

    def shrink_mask(self, pixels: int = 2) -> np.ndarray:
        """Mask 形态学腐蚀 (收缩)"""
        pixels = max(1, min(pixels, 20))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
        
        for z in range(self.shape[2]):
            if np.any(self.current_mask_3d[:, :, z]):
                self.current_mask_3d[:, :, z] = cv2.erode(self.current_mask_3d[:, :, z], kernel)
        return self.current_mask_3d

    def remove_artifacts(self, min_size: int = 50) -> np.ndarray:
        """清除孤立的杂质伪影 (去除小连通块)"""
        for z in range(self.shape[2]):
            mask_z = self.current_mask_3d[:, :, z]
            if np.any(mask_z):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_z)
                cleaned_z = np.zeros_like(mask_z)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= min_size:
                        cleaned_z[labels == i] = 255
                self.current_mask_3d[:, :, z] = cleaned_z
        return self.current_mask_3d

    def invert_mask(self) -> np.ndarray:
        """反选 Mask"""
        self.current_mask_3d = 255 - self.current_mask_3d
        return self.current_mask_3d

    def reset_mask(self) -> np.ndarray:
        """重置 Mask 为全空"""
        self.current_mask_3d = np.zeros(self.shape, dtype=np.uint8)
        return self.current_mask_3d

    def set_mask_3d(self, new_mask: np.ndarray):
        """用指定 3D 数组更新当前 Mask"""
        self.current_mask_3d = new_mask.copy()

    def get_mask_volume_raw_base64(self) -> dict:
        """
        导出当前 3D Mask 的 Uint8 体素数组 (与 get_volume_raw_base64 完全相同的轴变换)
        内存布局 (dz, dy, dx) C-order，与体数据坐标系 100% 一致
        """
        import cv2 as _cv2
        mask = self.current_mask_3d.astype(np.uint8)  # shape (dx, dy, dz)

        d_x, d_y, d_z = 96, 96, 96
        resized_mask = np.zeros((d_x, d_y, d_z), dtype=np.uint8)
        for zi in range(d_z):
            orig_z = int(zi * (self.shape[2] - 1) / (d_z - 1))
            slice_z = mask[:, :, orig_z]            # shape (182, 218)
            resized = _cv2.resize(slice_z, (d_y, d_x), interpolation=_cv2.INTER_NEAREST)
            resized_mask[:, :, zi] = resized

        # 与体数据完全相同的 transpose + flip
        mask_export = np.ascontiguousarray(
            np.flip(resized_mask.transpose(2, 1, 0), axis=1)
        )  # (dz, dy, dx)
        buffer_bytes = mask_export.tobytes()
        encoded = base64.b64encode(buffer_bytes).decode("utf-8")

        return {
            "dimensions": [d_z, d_y, d_x],
            "raw_base64": encoded,
            "has_mask": bool(np.any(mask > 0))
        }

    def export_nifti_mask(self, output_path: str) -> str:
        """将当前 3D Mask 保存导出为标准的 NIfTI 金标文件 (.nii.gz)"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_nii = nib.Nifti1Image(self.current_mask_3d.astype(np.uint8), affine=self.affine, header=self.header)
        nib.save(mask_nii, output_path)
        print(f"[Export] 金标已保存至: {output_path}")
        return output_path

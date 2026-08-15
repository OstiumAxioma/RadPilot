import os
import re
import json
import requests

class LLMRouter:
    def __init__(self, api_key_file: str = "api/gemini_testAPI.txt"):
        self.api_key = self._load_api_key(api_key_file)
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
        
    def _load_api_key(self, path: str) -> str:
        """从文件读取 API 密钥"""
        if not os.path.exists(path):
            print(f"[Warning] API Key 文件不存在: {path}")
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith("AQ."):
                        return line
                lines = [l.strip() for l in content.splitlines() if l.strip()]
                if lines:
                    return lines[-1]
        except Exception as e:
            print(f"[Warning] 读取 API Key 出错: {e}")
        return ""

    def parse_intent(self, prompt: str) -> dict:
        """
        解析医生的自然语言指令
        必须真实发起 HTTP 请求给 Gemini REST API
        """
        print(f"[LLMRouter] 真实发起 Gemini API 请求: '{prompt}'")
        
        # 1. 直接真实调用 Gemini API
        if self.api_key:
            llm_result = self._call_gemini_api(prompt)
            if llm_result:
                llm_result["source"] = "gemini_api_real"
                return llm_result

        # 2. 如果 API 未能成功返回 (如网络或配额限制)，尝试降级解析并提醒用户
        fallback = self._fast_rule_match(prompt)
        if fallback:
            fallback["source"] = "api_error_fallback"
            return fallback

        return {
            "action": "UNKNOWN",
            "pixels": 2,
            "explanation": "未能识别指令意图，请尝试说“生成全脑初标”、“外扩2像素”、“撤销”等。",
            "source": "fallback"
        }

    def _fast_rule_match(self, prompt: str) -> dict:
        """规则兜底匹配"""
        p = prompt.strip().lower()
        pixels_match = re.search(r'(\d+)\s*(像素|px|mm)?', p)
        pixels = int(pixels_match.group(1)) if pixels_match else 2
        
        if any(k in p for k in ["去颅骨", "脑实质", "全脑", "初标", "自动分割", "剥离颅骨"]):
            return {"action": "SKULL_STRIP", "pixels": 0, "explanation": "降级匹配: 全脑去颅骨脑实质提取"}
        if any(k in p for k in ["扩大", "外扩", "膨胀", "变大", "增加", "expand"]):
            return {"action": "EXPAND", "pixels": pixels, "explanation": f"降级匹配: Mask 外扩 {pixels} 像素"}
        if any(k in p for k in ["缩小", "收缩", "腐蚀", "变小", "减少", "shrink"]):
            return {"action": "SHRINK", "pixels": pixels, "explanation": f"降级匹配: Mask 收缩 {pixels} 像素"}
        if any(k in p for k in ["伪影", "杂质", "噪声", "杂点", "碎屑", "擦除", "过滤"]):
            return {"action": "REMOVE_ARTIFACTS", "min_size": 50, "explanation": "降级匹配: 过滤独立伪影"}
        if any(k in p for k in ["撤销", "上一步", "回退", "undo"]):
            return {"action": "UNDO", "explanation": "降级匹配: 撤销"}
        if any(k in p for k in ["重做", "下一步", "redo"]):
            return {"action": "REDO", "explanation": "降级匹配: 重做"}
        if any(k in p for k in ["导出", "保存", "金标", "nifti", "nii", "export"]):
            return {"action": "EXPORT", "explanation": "降级匹配: 导出金标"}
        if any(k in p for k in ["重置", "清空", "删除掩码", "reset"]):
            return {"action": "RESET", "explanation": "降级匹配: 重置"}
        return None

    def _call_gemini_api(self, prompt: str) -> dict:
        """真实调用 Gemini Flash REST API 进行结构化 JSON 输出"""
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        
        system_instruction = (
            "你是一个专业的医疗图像 Agent 助手 RadPilot。请分析放射科医生的自然语言指令，"
            "并严格只返回一个 JSON 对象，不要带有任何 Markdown 格式化或多余解释文字。\n"
            "JSON Schema 要求如下：\n"
            "{\n"
            '  "action": "SKULL_STRIP" | "EXPAND" | "SHRINK" | "REMOVE_ARTIFACTS" | "INVERT" | "RESET" | "UNDO" | "REDO" | "EXPORT" | "UNKNOWN",\n'
            '  "region": "full" | "left" | "right",\n'
            '  "pixels": 2,\n'
            '  "explanation": "你的思考与解析说明"\n'
            "}\n"
            "示例1：若医生输入'分割出左脑脑实质'，输出：{\"action\": \"SKULL_STRIP\", \"region\": \"left\", \"explanation\": \"通过 Gemini 成功识别意图：分割左半脑实质\"}。\n"
            "示例2：若医生输入'脑轮廓外扩3像素'，输出：{\"action\": \"EXPAND\", \"pixels\": 3, \"explanation\": \"将掩码外扩 3 像素\"}。"
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_instruction}\n\n医生自然语言指令: {prompt}"}
                    ]
                }
            ]
        }
        
        try:
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=10)
            print(f"[Gemini API HTTP Code]: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"[Gemini Raw Output]: {text}")
                
                # 过滤 Markdown 代码块标记 (```json ... ```)
                if "```" in text:
                    text = re.sub(r'^```(json)?\s*', '', text, flags=re.MULTILINE)
                    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
                    text = text.strip()
                    
                parsed = json.loads(text)
                return parsed
            else:
                print(f"[Warning] Gemini API HTTP 错误 {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[Error] 真实 Gemini API 请求异常: {e}")
            
        return None


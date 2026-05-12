import requests
import json
import base64
import io
import time
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class GeminiImageGenerationNode:
    """ComfyUI node for Gemini image generation via APImart"""
    
    def __init__(self):
        self.api_url = "https://api.apimart.ai/v1/images/generations"
        self.query_url = "https://api.apimart.ai/v1/tasks"
        self.poll_interval = 4  # 轮询间隔（秒）
        self.max_polls = 60  # 最多轮询次数
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "mode": (["text-to-image", "image-to-image"], {"default": "image-to-image"}),
                "api_key": ("STRING", {"multiline": False}),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                "n": (["1", "2", "4"], {"default": "1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "size": (["1:1", "2:3", "3:2", "3:4", "4:3", "16:9"], {"default": "3:2"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "response")
    FUNCTION = "generate"
    CATEGORY = "image/generation"
    
    def tensor_to_base64(self, tensor) -> str:
        """Convert ComfyUI tensor image to base64 string"""
        # Convert tensor to NumPy array
        if torch is not None and isinstance(tensor, torch.Tensor):
            img_array = tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.ndarray):
            img_array = tensor
        else:
            try:
                img_array = np.array(tensor)
            except Exception as e:
                raise ValueError(f"Cannot convert input to array: {e}")
        
        # Handle batch dimension if present
        if img_array.ndim == 4:
            # Shape is (batch, height, width, channels)
            # Take the first image from batch
            img_array = img_array[0]
        
        # Now handle (height, width, channels)
        if img_array.ndim == 3:
            # Normalize to 0-255 range
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
            
            # Handle different channel counts
            if img_array.shape[2] == 3:  # RGB
                img = Image.fromarray(img_array, mode='RGB')
            elif img_array.shape[2] == 4:  # RGBA
                img = Image.fromarray(img_array, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
        else:
            raise ValueError(f"Unexpected tensor shape: {img_array.shape}")
        
        # Convert PIL Image to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_data}"
    
    def base64_to_tensor(self, base64_str: str) -> np.ndarray:
        """Convert base64 string to ComfyUI tensor"""
        # Remove data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to tensor (normalize to 0-1)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    
    def collect_images(self, **kwargs) -> List[str]:
        """Collect optional input images and convert to base64"""
        image_urls = []
        for i in range(1, 11):
            key = f"image_{i}"
            if key in kwargs and kwargs[key] is not None:
                base64_str = self.tensor_to_base64(kwargs[key])
                image_urls.append(base64_str)
        return image_urls
    
    def poll_task_status(self, task_id: str, api_key: str) -> Tuple[str, dict]:
        """轮询任务状态直到完成"""
        print(f"[GeminiNode] 步骤 4: 轮询任务状态 (每 {self.poll_interval} 秒查询一次，共 {self.max_polls} 次)...")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        for poll_count in range(1, self.max_polls + 1):
            try:
                # 查询任务状态
                query_endpoint = f"{self.query_url}/{task_id}"
                response = requests.get(query_endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                
                response_data = response.json()
                print(f"[GeminiNode] 轮询 {poll_count}/{self.max_polls}: {json.dumps(response_data, ensure_ascii=False)}")
                
                # 检查任务状态
                if response_data.get("code") == 200:
                    data = response_data.get("data", {})
                    status = data.get("status")
                    
                    if status == "succeeded" or status == "completed":
                        # 任务完成 - 处理新的响应结构
                        result = data.get("result", {})
                        images = result.get("images", [])
                        
                        if images and len(images) > 0:
                            # 获取第一张图片的 URL
                            image_info = images[0]
                            url_list = image_info.get("url", [])
                            
                            # url 可能是列表或字符串
                            if isinstance(url_list, list) and len(url_list) > 0:
                                image_url = url_list[0]
                            elif isinstance(url_list, str):
                                image_url = url_list
                            else:
                                raise Exception("Invalid image URL format in response")
                            
                            print(f"[GeminiNode] 任务完成，结果 URL: {image_url}")
                            return image_url, response_data
                        else:
                            raise Exception("Task completed but no images returned")
                    
                    elif status == "failed":
                        error_msg = data.get("error", "Unknown error")
                        raise Exception(f"Task failed: {error_msg}")
                    
                    elif status == "processing":
                        print(f"[GeminiNode] 任务处理中... ({poll_count}/{self.max_polls})")
                        time.sleep(self.poll_interval)
                    
                    else:
                        print(f"[GeminiNode] 未知状态: {status}，继续轮询...")
                        time.sleep(self.poll_interval)
                else:
                    raise Exception(f"API returned code {response_data.get('code')}: {response_data}")
            
            except requests.exceptions.RequestException as e:
                print(f"[GeminiNode] 轮询请求失败: {str(e)}")
                if poll_count < self.max_polls:
                    time.sleep(self.poll_interval)
                else:
                    raise Exception(f"Failed to query task after {self.max_polls} attempts: {str(e)}")
        
        raise Exception(f"Task did not complete within {self.max_polls * self.poll_interval} seconds")
    
    def download_image(self, image_url: str) -> Image.Image:
        """下载图片"""
        print(f"[GeminiNode] 步骤 5: 下载结果图片...")
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            print(f"[GeminiNode] 图片下载成功，大小: {img.size}")
            return img
        except Exception as e:
            raise Exception(f"Failed to download image: {str(e)}")
    
    def generate(
        self,
        mode: str,
        api_key: str,
        prompt: str,
        model: str,
        n: str,
        resolution: str,
        size: str,
        seed: int,
        **kwargs
    ) -> Tuple[Any, str, str]:
        """Generate images using Gemini API"""
        
        try:
            print(f"\n[GeminiNode] 开始生成图片...")
            print(f"[GeminiNode] 模式: {mode}")
            print(f"[GeminiNode] 提示词: {prompt}")
            print(f"[GeminiNode] 模型: {model}")
            
            # 步骤 1-2: 处理图片输入
            image_urls = []
            if mode == "image-to-image":
                print("[GeminiNode] 步骤 1: 转换图片格式...")
                image_urls = self.collect_images(**kwargs)
                if not image_urls:
                    raise Exception("图生图模式必须提供至少一张输入图片")
                print(f"[GeminiNode] 步骤 2: 成功转换 {len(image_urls)} 张图片")
            elif mode == "text-to-image":
                print("[GeminiNode] 文生图模式，跳过图片转换步骤")
            else:
                raise Exception(f"未知的生成模式: {mode}")
            
            # 步骤 3: 调用生成接口
            print(f"[GeminiNode] 步骤 {3 if mode == 'image-to-image' else 1}: 调用图生接口...")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": int(n),
                "resolution": resolution,
            }
            
            # Add seed if provided
            if seed > 0:
                payload["seed"] = seed
            
            # Add reference images for image-to-image mode
            if mode == "image-to-image" and image_urls:
                payload["image_urls"] = image_urls
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print(f"[GeminiNode] 发送请求到: {self.api_url}")
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            print(f"[GeminiNode] API 响应: {json.dumps(response_data, ensure_ascii=False)}")
            
            # Check for API errors
            if "error" in response_data:
                error_msg = response_data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"API Error: {error_msg}")
            
            # Extract task_id from response
            if response_data.get("code") != 200 or not response_data.get("data"):
                raise Exception(f"Invalid API response: {response_data}")
            
            task_id = response_data["data"][0].get("task_id")
            print(f"[GeminiNode] 任务已创建，ID: {task_id}")
            
            # 步骤 4: 轮询任务状态
            result_image_url, final_response = self.poll_task_status(task_id, api_key)
            
            # 步骤 5: 下载结果图片
            result_image_pil = self.download_image(result_image_url)
            
            # 步骤 6: 转换回 Tensor
            print(f"[GeminiNode] 步骤 {6 if mode == 'image-to-image' else 4}: 转换回 ComfyUI 格式...")
            result_tensor = self.pil_to_tensor(result_image_pil)
            
            response_text = json.dumps(final_response, ensure_ascii=False, indent=2)
            
            print("[GeminiNode] 处理完成!\n")
            return (result_tensor, result_image_url, response_text)
        
        except Exception as e:
            print(f"[GeminiNode] 错误: {str(e)}\n")
            raise Exception(f"GeminiNode 执行失败: {str(e)}")
    
    def pil_to_tensor(self, image: Image.Image):
        """Convert PIL Image to ComfyUI tensor"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize to 0-1
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # [1, height, width, 3]
        
        # Convert to PyTorch tensor if available
        if torch is not None:
            img_tensor = torch.from_numpy(img_array)
        else:
            img_tensor = img_array
        
        return img_tensor


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerationNode": GeminiImageGenerationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerationNode": "Gemini Image Generation (APImart)",
}

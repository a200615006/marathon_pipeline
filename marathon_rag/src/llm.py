from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any
import requests
from datetime import datetime
import os
from .config import API_KEY, MAX_TOKENS, TEMPERATURE
import logging

# 基本配置
logging.basicConfig(level=logging.INFO)
log = logging.info  # 或者使用 logger
class SiliconFlowLLM(CustomLLM):
    """硅基流动自定义 LLM。"""
    
    model: str = "Qwen/Qwen3-32B"
    api_key: str = API_KEY
    api_base: str = "https://api.siliconflow.cn/v1"
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    
    save_requests: bool = True
    save_dir: str = "./logs"
    save_filename: str = "requests_log.txt"
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=32768,
            num_output=self.max_tokens,
            model_name=self.model,
        )
    
    def _save_request(self, prompt: str, response_text: str = None):
        if not self.save_requests:
            return
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            filepath = os.path.join(self.save_dir, self.save_filename)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            separator = "=" * 80
            content = f"\n{separator}\n时间: {timestamp}\n模型: {self.model}\n{separator}\n【请求内容】\n{prompt}\n"
            if response_text:
                content += f"\n【响应内容】\n{response_text}\n"
            content += f"{separator}\n"
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(content)
            log(f"✓ 请求已保存到: {filepath}")
        except Exception as e:
            log(f"✗ 保存请求失败: {str(e)}")
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
            "enable_thinking":False
        }
        response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        self._save_request(prompt, response_text)
        return CompletionResponse(text=response_text)
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        yield response
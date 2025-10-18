import json
import logging
import os
import requests
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI

from calculate import CalculatorTool
from currentDateTool import CurrentDateTool
from nl2sql_call import Nl2sqlTool
from req_resp_obj import ToolResponse, QueryResponse
from config import MODEL_NAME, PROMPT_CHOICE, PROMPT_QA, MAIN_LOG_FILE

# 配置日志
logging.basicConfig(
    filename=MAIN_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)
# 测试日志
logger.info("=== 应用程序启动 ===")


class AdvancedMCPHttpToolManager:
    def __init__(self, api_key: str, base_url: str = None, tools_directory: str = "mcp_tools", max_iterations: int = 5,
                 headers: Dict[str, str] = None):
        """
        高级MCP HTTP工具管理器，支持多次调用和多个工具

        Args:
            api_key: OpenAI API密钥
            base_url: OpenAI API基础URL
            tools_directory: MCP工具描述文件目录
            max_iterations: 最大迭代次数，防止无限循环
            headers: HTTP请求头
        """
        client_args = {"api_key": api_key}
        if base_url:
            client_args["base_url"] = base_url
            client_args["timeout"] = 60.0

        self.client = OpenAI(**client_args)
        self.tools_directory = tools_directory
        self.max_iterations = max_iterations

        # todo
        # 先初始化本地工具
        self.local_tools = {
            "calculator_tool": CalculatorTool.calculate,
            "current_date_tool": CurrentDateTool.get_current_date,
            "nl2sql_tool": Nl2sqlTool.call
        }

        # 然后加载工具文件
        self.tools = self.load_tools_from_files()

        self.conversation_history = []
        self.call_log = []  # 记录所有工具调用
        self.headers = headers or {}

        print(f"📊 总共加载了 {len(self.tools)} 个工具")
        print(f"🔧 本地工具: {list(self.local_tools.keys())}")

        # print(f"🔧 本地工具: {json.dumps(self.local_tools.values(), indent=4, ensure_ascii=False)}")
        # print(f"🔧 HTTP工具: {json.dumps(self.tools, indent=4, ensure_ascii=False)}")

    def load_tools_from_files(self) -> List[Dict[str, Any]]:
        """从文本文件加载MCP工具描述"""
        tools = []

        if not os.path.exists(self.tools_directory):
            print(f"⚠️ 工具目录不存在: {self.tools_directory}")
            return tools

        for filename in os.listdir(self.tools_directory):
            if filename.endswith(('.txt', '.json')):
                file_path = os.path.join(self.tools_directory, filename)
                try:
                    tool_config = self.parse_tool_file(file_path)
                    if tool_config:
                        # 检查是否为本地工具
                        tool_name = tool_config['function']['name']
                        if tool_name in self.local_tools:
                            print(f"🔧 本地工具: {tool_name}")
                            # 对于本地工具，移除http_config
                            if 'http_config' in tool_config:
                                del tool_config['http_config']
                        else:
                            print(f"🌐 HTTP工具: {tool_name}")

                        tools.append(tool_config)
                except Exception as e:
                    print(f"❌ 加载工具文件 {filename} 时出错: {e}")

        return tools

    def parse_tool_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """解析单个工具描述文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # 支持JSON格式
        if content.startswith('{'):
            try:
                tool_data = json.loads(content)
                # 检查是否为本地工具
                tool_name = tool_data.get('name', '')
                if tool_name not in self.local_tools:
                    # 只有非本地工具才需要http_config
                    if 'http_config' not in tool_data:
                        tool_data['http_config'] = {
                            'url': tool_data.get('url', ''),
                            'method': tool_data.get('method', 'GET')
                        }
                return tool_data
            except json.JSONDecodeError as e:
                print(f"JSON解析错误 {file_path}: {e}")
                return None

        # 解析文本格式
        return self.parse_text_format(content, os.path.basename(file_path))

    def parse_text_format(self, content: str, filename: str) -> Dict[str, Any]:
        """解析文本格式的工具描述"""
        lines = content.split('\n')
        tool_info = {
            "name": "",
            "description": "",
            "http_config": {},
            "parameters": {"type": "object", "properties": {}, "required": []}
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            if line.lower().startswith('name:'):
                tool_info['name'] = line[5:].strip()
            elif line.lower().startswith('description:'):
                tool_info['description'] = line[12:].strip()
            elif line.lower().startswith('http_url:'):
                tool_info['http_config']['url'] = line[9:].strip()
            elif line.lower().startswith('http_method:'):
                tool_info['http_config']['method'] = line[12:].strip().upper()
            elif line.lower().startswith('parameters:'):
                current_section = 'parameters'
            elif line.lower().startswith('required:'):
                required_params = line[9:].strip().split(',')
                tool_info['parameters']['required'] = [p.strip() for p in required_params if p.strip()]
            elif ':' in line and current_section == 'parameters':
                param_name, param_desc = line.split(':', 1)
                param_name = param_name.strip()
                param_desc = param_desc.strip()

                param_type = "string"
                if any(word in param_desc.lower() for word in ['number', 'int', 'float', 'integer']):
                    param_type = "number"
                elif any(word in param_desc.lower() for word in ['boolean', 'bool']):
                    param_type = "boolean"
                elif any(word in param_desc.lower() for word in ['array', 'list']):
                    param_type = "array"

                tool_info['parameters']['properties'][param_name] = {
                    "type": param_type,
                    "description": param_desc
                }

        if not tool_info['name']:
            tool_info['name'] = os.path.splitext(filename)[0]

        # 检查是否为本地工具
        if tool_info['name'] in self.local_tools:
            # 本地工具不需要http_config
            if 'http_config' in tool_info:
                del tool_info['http_config']
        else:
            # HTTP工具需要method
            if 'method' not in tool_info['http_config']:
                tool_info['http_config']['method'] = 'GET'

        return {
            "type": "function",
            "function": {
                "name": tool_info['name'],
                "description": tool_info['description'],
                "parameters": tool_info['parameters']
            },
            "http_config": tool_info.get('http_config', {})
        }

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResponse:
        """调用工具并返回标准化响应"""
        call_start = time.time()

        try:
            # 检查是否为本地工具
            if tool_name in self.local_tools:
                print(f"🔧 调用本地工具: {tool_name}")
                print(f"📤 请求参数: {arguments}")

                # 调用本地工具方法
                local_method = self.local_tools[tool_name]
                result = local_method(**arguments)

                call_duration = time.time() - call_start

                # 记录调用日志
                call_log = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "status_code": 200,  # 本地工具总是成功
                    "duration": round(call_duration, 2),
                    "timestamp": datetime.now().isoformat(),
                    "type": "local"
                }
                self.call_log.append(call_log)

                # 处理结果格式
                if isinstance(result, dict) and 'success' in result:
                    success = result['success']
                    result_str = result['result'] if 'result' in result else str(result)
                else:
                    success = True
                    result_str = str(result)

                print(f"✅ 工具调用成功 (耗时: {call_duration:.2f}s)")
                return ToolResponse(
                    success=success,
                    result=result_str,
                    tool_name=tool_name,
                    duration=call_duration
                )
                # return result_str;

            else:
                tool_config = None
                for tool in self.tools:
                    if tool['function']['name'] == tool_name and 'http_config' in tool:
                        tool_config = tool
                        break

                if not tool_config:
                    error_msg = f"未找到工具 '{tool_name}' 的配置"
                    return ToolResponse(
                        success=False,
                        result=error_msg,
                        tool_name=tool_name,
                        duration=time.time() - call_start
                    )

                http_config = tool_config['http_config']
                url = http_config.get('url')
                method = http_config.get('method', 'GET')

                if not url:
                    error_msg = f"工具 '{tool_name}' 未配置HTTP URL"
                    return ToolResponse(
                        success=False,
                        result=error_msg,
                        tool_name=tool_name,
                        duration=time.time() - call_start
                    )

                print(f"🌐 调用HTTP服务: {method} {url}")
                print(f"📤 请求参数: {arguments}")

                # 准备请求头
                headers = {**self.headers}
                if method in ['POST', 'PUT'] and 'Content-Type' not in headers:
                    headers['Content-Type'] = 'application/json'

                # 根据HTTP方法调用服务
                if method == 'GET':
                    response = requests.get(url, params=arguments, headers=headers, timeout=30)
                elif method == 'POST':
                    response = requests.post(url, json=arguments, headers=headers, timeout=30)
                elif method == 'PUT':
                    response = requests.put(url, json=arguments, headers=headers, timeout=30)
                elif method == 'DELETE':
                    response = requests.delete(url, params=arguments, headers=headers, timeout=30)
                else:
                    error_msg = f"不支持的HTTP方法 {method}"
                    return ToolResponse(
                        success=False,
                        result=error_msg,
                        tool_name=tool_name,
                        duration=time.time() - call_start
                    )

                call_duration = time.time() - call_start

                # 记录调用日志
                call_log = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "status_code": response.status_code,
                    "duration": round(call_duration, 2),
                    "timestamp": datetime.now().isoformat(),
                    "type": "http"
                }
                self.call_log.append(call_log)

                if response.status_code == 200:
                    result = response.json() if 'application/json' in response.headers.get('content-type',
                                                                                           '') else response.text
                    print(f"✅ HTTP调用成功 (耗时: {call_duration:.2f}s)")
                    return ToolResponse(
                        success=True,
                        result=str(result),
                        tool_name=tool_name,
                        duration=call_duration
                    )
                else:
                    error_msg = f"HTTP错误 {response.status_code}: {response.text}"
                    print(f"❌ {error_msg}")
                    return ToolResponse(
                        success=False,
                        result=error_msg,
                        tool_name=tool_name,
                        duration=call_duration
                    )

        except Exception as e:
            error_msg = f"工具调用异常: {str(e)}"
            call_duration = time.time() - call_start
            print(f"🚨 {error_msg}")
            return ToolResponse(
                success=False,
                result=error_msg,
                tool_name=tool_name,
                duration=call_duration
            )

    def process_user_query(self, question: str, content: str) -> QueryResponse:
        """
        处理用户查询，支持多次工具调用和多个工具
        Args:
            question: 用户查询
            content: 选项
        Returns: QueryResponse: 处理结果
        """
        max_iterations = self.max_iterations

        self.conversation_history = []

        if content:
            messages = self.conversation_history + PROMPT_CHOICE + [{"role": "user", "content": question + " \n" + content}]
        else:
            messages = self.conversation_history + PROMPT_QA + [{"role": "user", "content": question}]
        # print(f"messages={messages} | question={question}")

        iteration_count = 0
        tool_call_count = 0
        tool_calls_info = []

        print(f"\n##############################################################")
        print(f"🔍 \n开始处理查询: {question}")

        while tool_call_count < max_iterations:
            iteration_count += 1
            print(f"\n🔄 第 {iteration_count} 轮处理")

            try:
                # 准备工具列表（移除http_config）
                available_tools = [{k: v for k, v in tool.items() if k != 'http_config'} for tool in self.tools]

                print(f"iteration_count={iteration_count} | messages={messages} | question={question}")

                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=available_tools if available_tools else None,
                    tool_choice="auto" if available_tools else "none",
                    timeout=30.0,
                    extra_body={
                        "enable_thinking": False  # 禁用思考过程
                    }
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                final_reply = response_message.content
                # 如果没有工具调用，直接返回结果
                if not tool_calls:
                    print(f"💬 无可用工具调用 | 模型选择直接回复 (第{iteration_count}轮)")

                    # 更新对话历史
                    self.conversation_history.extend([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": final_reply}
                    ])

                    if tool_call_count == 0:
                        return QueryResponse(
                            code="1",
                            success=True,
                            response=final_reply,
                            tool_calls=tool_calls_info,
                            total_iterations=iteration_count
                        )
                    else:
                        return QueryResponse(
                            code="0",
                            success=True,
                            response=final_reply,
                            tool_calls=tool_calls_info,
                            total_iterations=iteration_count
                        )

                # 处理工具调用
                tool_call_count += 1

                print(f"🔧 (第{tool_call_count} 轮工具调用）| 模型决定调用 {len(tool_calls)} 个工具")
                messages.append(response_message)

                # 执行所有工具调用
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🛠️ ##调用工具 [{function_name}]: {function_args}")

                    if function_name == "nl2sql_tool":
                        function_args = {
                            "query": question
                        }
                        tool_result = self.call_tool(function_name, function_args)
                        return QueryResponse(
                            code="0",
                            success=True,
                            response=tool_result.result,
                            tool_calls=tool_calls_info,
                            total_iterations=iteration_count
                        )

                    tool_result = self.call_tool(function_name, function_args)

                    print(f"tool_result={tool_result}")
                    # 记录工具调用信息
                    tool_calls_info.append({
                        "tool_name": function_name,
                        "arguments": function_args,
                        "success": tool_result.success,
                        "duration": tool_result.duration,
                        "type": "local" if function_name in self.local_tools else "http"
                    })

                    # 将工具结果添加到消息中
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result.result,
                    })

                # 检查是否应该继续迭代
                if tool_call_count >= max_iterations:
                    print("⚠️ 达到最大工具调用次数，生成最终回复")
                    break

            except Exception as e:
                error_msg = f"第 {iteration_count} 轮处理时出错: {e}"
                print(f"🚨 {error_msg}")
                messages.append({"role": "system", "content": f"处理错误: {e}"})
                break

        # 生成最终回复
        try:
            print(f"final | iteration_count={iteration_count} | messages={messages} | question={question}")
            final_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=30.0,
                extra_body={
                    "enable_thinking": False  # 禁用思考过程
                }
            )

            final_content = final_response.choices[0].message.content
            print(f"✅ 处理完成，共进行 {iteration_count} 轮，调用 {len(tool_calls_info)} 次工具")

            # 更新对话历史
            self.conversation_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_content}
            ])

            return QueryResponse(
                success=True,
                response=final_content,
                tool_calls=tool_calls_info,
                total_iterations=iteration_count
            )

        except Exception as e:
            error_msg = f"生成最终回复时出错: {e}"
            return QueryResponse(
                success=False,
                response=error_msg,
                tool_calls=tool_calls_info,
                total_iterations=iteration_count
            )

    def get_call_statistics(self) -> Dict[str, Any]:
        """获取工具调用统计信息"""
        if not self.call_log:
            return {"total_calls": 0, "tools_used": []}

        tool_usage = {}
        local_calls = 0
        http_calls = 0

        for call in self.call_log:
            tool_name = call["tool"]
            call_type = call.get("type", "http")

            if tool_name in tool_usage:
                tool_usage[tool_name] += 1
            else:
                tool_usage[tool_name] = 1

            if call_type == "local":
                local_calls += 1
            else:
                http_calls += 1

        return {
            "total_calls": len(self.call_log),
            "unique_tools": len(tool_usage),
            "tools_used": tool_usage,
            "local_calls": local_calls,
            "http_calls": http_calls,
            "average_duration": round(sum(call["duration"] for call in self.call_log) / len(self.call_log),
                                      2) if self.call_log else 0
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        return [
            {
                "name": tool['function']['name'],
                "description": tool['function']['description'],
                "parameters": tool['function']['parameters'],
                "type": "local" if tool['function']['name'] in self.local_tools else "http"
            }
            for tool in self.tools
        ]

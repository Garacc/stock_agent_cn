import os
import time
from google import genai
from openai import OpenAI  # 更新 OpenAI 导入方式
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from src.utils.logger_config import get_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# GLOBAL SETTINGS
# 设置日志记录
logger = get_logger()

@dataclass
class ChatMessage:
    content: str

@dataclass
class ChatChoice:
    message: ChatMessage

@dataclass
class ChatCompletion:
    choices: list[ChatChoice]

# 模型处理器配置字典
model_handlers = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "env_model": "GEMINI_MODEL",
        "default_model": "gemini-1.5-flash",
        "init_func": lambda key: genai.Client(api_key=key),
        "name": "Gemini"
    },
    "moonshot": {
        "env_key": "KIMI_API_KEY",
        "env_model": "KIMI_MODEL",
        "default_model": "moonshot-v1-8k",
        "init_func": lambda key: OpenAI(api_key=key, base_url="https://api.moonshot.cn/v1"),
        "name": "Moonshot"
    }
}

class ClientManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
            cls._instance.clients = {}  # 初始化客户端字典
            # 获取项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_path = os.path.join(project_root, '.env')

            # 加载环境变量
            if os.path.exists(env_path):
                load_dotenv(env_path, override=True)
                logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
            else:
                logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")
        
        return cls._instance
    
    def get_clients_info(self, models):
        """获取模型客户端信息，如果客户端已初始化则复用"""
        if models is None:
            # 如果未指定模型，默认使用所有可用模型
            models = list(model_handlers.keys())
            logger.info(f"未指定模型，将尝试使用所有可用模型: {models}")
        
        if isinstance(models, str):
            models = [models]
            
        for model in models:
            if model in self.clients.keys():
                logger.info(f"{SUCCESS_ICON} {model} 客户端已被初始化")
                continue
                
            if model in model_handlers.keys():
                handler = model_handlers[model]
                env_key = os.getenv(handler["env_key"])
                env_model = os.getenv(handler["env_model"]) if os.getenv(handler["env_model"]) else handler["default_model"]
                init_func = handler["init_func"]
                name = handler["name"]
                
                if env_key and env_model:
                    logger.info(f"{SUCCESS_ICON} {name} 客户端初始化成功，当前选用模型为: {env_model}")
                    client = init_func(env_key)
                    self.clients[model] = (client, env_model)
                else:
                    logger.warning(f"{ERROR_ICON} {name} 客户端初始化失败，未找到环境变量")
            else:
                logger.warning(f"{ERROR_ICON} 未知模型: {model}")
        
        logger.info(f"已初始化 {len(self.clients)} 个客户端")
        return self.clients

# 创建全局的客户端管理器实例
client_manager = ClientManager()

@backoff.on_exception(
    backoff.expo,
    (Exception),  # 使用通用异常，因为新版 OpenAI 客户端异常类型可能不同
    max_tries=5,
    max_time=300
)
def generate_openai_content_with_retry(client, model, messages):
    """带重试机制的基于OpenAI公共API的内容生成函数"""
    try:
        if client is None:
            raise ValueError("OpenAI客户端未初始化")
        
        logger.info(f"{WAIT_ICON} 正在调用 {model} ...")
        logger.info(f"请求消息: {str(messages)[:500]}..." if len(str(messages)) > 500 else f"请求消息: {messages}")

        # 使用 OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,  # Kimi 特定参数
        )

        logger.info(f"{SUCCESS_ICON} {model} 调用成功")
        content = response.choices[0].message.content
        logger.info(f"响应内容: {content[:500]}..." if len(content) > 500 else f"响应内容: {content}")
        return response
    except Exception as e:
        logger.error(f"{ERROR_ICON} {model} 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e

@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_google_content_with_retry(client, model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        if client is None:
            raise ValueError("GenAI客户端未初始化")
        logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
        logger.info(f"请求内容: {contents[:500]}..." if len(
            str(contents)) > 500 else f"请求内容: {contents}")
        logger.info(f"请求配置: {config}")

        response = client.models.generate_content(  # 使用 gemini_client
            model=model,
            contents=contents,
            config=config
        )

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.info(f"响应内容: {response.text[:500]}..." if len(
            str(response.text)) > 500 else f"响应内容: {response.text}")
        return response
    except Exception as e:
        if "AFC is enabled" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e

def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """获取聊天完成的内容，支持多种模型以及同时调用多个模型"""
    clients = client_manager.get_clients_info(model)
    contents = {}
    
    if not clients:
        logger.error(f"{ERROR_ICON} 没有可用的客户端")
        return contents
    
    for k, v in clients.items():
        try:
            logger.info(f"{WAIT_ICON} 使用模型: {k}")
            logger.debug(f"消息内容: {messages}")
            
            is_gemini = "gemini" in k.lower()
            _client, _env_model = v

            # 确保客户端已初始化
            if _client is None:
                logger.error(f"{ERROR_ICON} {k} 客户端未初始化")
                continue

            for attempt in range(max_retries):
                try:
                    if not is_gemini:   # 非 Gemini 模型统一使用OpenAI API
                        # 直接调用 OpenAI API
                        response = generate_openai_content_with_retry(
                            client=_client,
                            model=_env_model,
                            messages=messages,
                        )
                        if response is None:
                            raise ValueError(f"{k} API 返回空值")
                        content = response.choices[0].message.content
                    else:               # Gemini API 调用前需要转换格式
                        prompt = ""
                        system_instruction = None

                        for message in messages:
                            role = message["role"]
                            content = message["content"]
                            if role == "system":
                                system_instruction = content
                            elif role == "user":
                                prompt += f"User: {content}\n"
                            elif role == "assistant":
                                prompt += f"Assistant: {content}\n"

                        config = {}
                        if system_instruction:
                            config['system_instruction'] = system_instruction

                        response = generate_google_content_with_retry(
                            client=_client,
                            model=_env_model,
                            contents=prompt.strip(),
                            config=config
                        )
                        
                        if response is None:
                            raise ValueError(f"{k} API 返回空值")
                        
                        content = response.text

                    logger.info(f"{SUCCESS_ICON} {k} 成功获取响应")
                    logger.debug(f"原始响应: {content[:500]}..." if len(
                        content) > 500 else f"原始响应: {content}")
                    contents[k] = content
                    break
                
                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} {k} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} {k} 最终错误: {str(e)}")

        except Exception as e:
            logger.error(f"{ERROR_ICON} {k} 处理过程中发生错误: {str(e)}")
            continue

    if not contents:
        logger.error(f"{ERROR_ICON} 所有模型调用均失败")
    elif len(contents) < len(clients):
        logger.warning(f"{WAIT_ICON} 部分模型调用失败，成功率: {len(contents)}/{len(clients)}")
        
    return contents

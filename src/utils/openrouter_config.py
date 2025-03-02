import os
import time
from google import genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff
from openai import OpenAI  # 更新 OpenAI 导入方式
from src.utils.logger_config import setup_api_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

# 设置日志记录
logger = setup_api_logger()


@dataclass
class ChatMessage:
    content: str


@dataclass
class ChatChoice:
    message: ChatMessage


@dataclass
class ChatCompletion:
    choices: list[ChatChoice]


# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证 Gemini 环境变量
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

if not api_key:
    logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
    raise ValueError("GEMINI_API_KEY not found in environment variables")
if not model:
    model = "gemini-1.5-flash"
    logger.info(f"{WAIT_ICON} 使用默认模型: {model}")

# 初始化 Gemini 客户端
gemini_client = genai.Client(api_key=api_key)  # 重命名为 gemini_client
logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

# 验证 OpenAI 环境变量
# 环境变量验证部分添加 Kimi 配置
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL")
kimi_api_base = "https://api.moonshot.cn/v1"  # Kimi API 基础地址
openai_client = None

if not openai_api_key:
    logger.warning(f"{ERROR_ICON} 未找到 OPENAI_API_KEY 环境变量，OpenAI/Kimi 功能将不可用")
else:
    # 初始化 OpenAI 客户端 (使用 Kimi API)
    openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=kimi_api_base
    )
    if not openai_model:
        openai_model = "moonshot-v1-8k"  # 使用 Kimi 默认模型
        logger.info(f"{WAIT_ICON} 使用默认 Kimi 模型: {openai_model}")
    logger.info(f"{SUCCESS_ICON} Kimi 客户端初始化成功")


@backoff.on_exception(
    backoff.expo,
    (Exception),
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """带重试机制的内容生成函数"""
    try:
        logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
        logger.info(f"请求内容: {contents[:500]}..." if len(
            str(contents)) > 500 else f"请求内容: {contents}")
        logger.info(f"请求配置: {config}")

        response = gemini_client.models.generate_content(  # 使用 gemini_client
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


@backoff.on_exception(
    backoff.expo,
    (Exception),  # 使用通用异常，因为新版 OpenAI 客户端异常类型可能不同
    max_tries=5,
    max_time=300
)
def generate_openai_content_with_retry(model, messages):
    """带重试机制的 Kimi/OpenAI 内容生成函数"""
    try:
        if openai_client is None:
            raise ValueError("Kimi 客户端未初始化")
            
        logger.info(f"{WAIT_ICON} 正在调用 Kimi API...")
        logger.info(f"请求消息: {str(messages)[:500]}..." if len(
            str(messages)) > 500 else f"请求消息: {messages}")

        # 使用 Kimi API
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,  # Kimi 特定参数
            top_p=0.95       # Kimi 特定参数
        )

        logger.info(f"{SUCCESS_ICON} Kimi API 调用成功")
        content = response.choices[0].message.content
        logger.info(f"响应内容: {content[:500]}..." if len(
            content) > 500 else f"响应内容: {content}")
        return response
    except Exception as e:
        logger.error(f"{ERROR_ICON} Kimi API 调用失败: {str(e)}")
        logger.error(f"错误详情: {str(e)}")
        raise e


def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    try:
        if model is None:
            if openai_client is not None:
                model = os.getenv("OPENAI_MODEL", "moonshot-v1-8k")  # 默认使用 Kimi 模型
            else:
                model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.debug(f"消息内容: {messages}")

        # 修改模型判断逻辑，添加 Kimi 模型前缀
        is_openai = model.startswith(("gpt-", "text-", "moonshot-"))
        
        for attempt in range(max_retries):
            try:
                if is_openai:
                    if openai_client is None:
                        logger.error(f"{ERROR_ICON} OpenAI 客户端未初始化")
                        return None
                    
                    # 直接调用 OpenAI API
                    response = generate_openai_content_with_retry(
                        model=model,
                        messages=messages
                    )
                    
                    if response is None:
                        raise ValueError("OpenAI API 返回空值")
                    
                    content = response.choices[0].message.content
                    
                else:
                    # Gemini API 调用前需要转换格式
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

                    response = generate_content_with_retry(
                        model=model,
                        contents=prompt.strip(),
                        config=config
                    )
                    
                    if response is None:
                        raise ValueError("Gemini API 返回空值")
                    
                    content = response.text

                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                logger.debug(f"原始响应: {content[:500]}..." if len(
                    content) > 500 else f"原始响应: {content}")
                return content

            except Exception as e:
                logger.error(
                    f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
        return None


# 删除不再需要的 get_openai_chat_completion 函数

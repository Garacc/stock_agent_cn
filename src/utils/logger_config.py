import os
import time
import logging

# 状态图标
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON = "⟳"

def setup_api_logger():
    """设置API调用日志记录器"""
    # 创建logger
    logger = logging.getLogger('api_calls')
    logger.setLevel(logging.DEBUG)

    # 移除所有现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 设置文件处理器
    log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
    print(f"Creating log file at: {log_file}")

    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        print("Successfully created file handler")
    except Exception as e:
        print(f"Error creating file handler: {str(e)}")
        return None

    # 设置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 立即测试日志记录
    logger.debug("Logger initialization completed")
    logger.info("API logging system started")

    return logger
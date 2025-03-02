import unittest
from unittest.mock import Mock, patch
import os
import sys
from src.utils.openrouter_config import get_chat_completion, ClientManager, model_handlers

class TestGetChatCompletion(unittest.TestCase):
    def setUp(self):
        """测试前的准备工作"""
        # 模拟消息
        self.messages = [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好!我是助手"}
        ]
        
        # 模拟成功的响应
        self.mock_response_openai = Mock()
        self.mock_response_openai.choices = [Mock(message=Mock(content="OpenAI的回复"))]
        
        self.mock_response_gemini = Mock()
        self.mock_response_gemini.text = "Gemini的回复"
        
        # 模拟客户端
        self.mock_openai_client = Mock()
        self.mock_gemini_client = Mock()

    @patch('src.utils.openrouter_config.client_manager')
    @patch('src.utils.openrouter_config.generate_openai_content_with_retry')
    def test_openai_model_success(self, mock_generate_openai, mock_client_manager):
        """测试OpenAI模型成功调用的情况"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {
            "moonshot": (self.mock_openai_client, "moonshot-v1-8k")
        }
        mock_generate_openai.return_value = self.mock_response_openai
        
        # 执行测试
        result = get_chat_completion(self.messages, model="moonshot")
        
        # 验证结果
        self.assertEqual(result, {"moonshot": "OpenAI的回复"})
        mock_generate_openai.assert_called_once()

    @patch('src.utils.openrouter_config.client_manager')
    @patch('src.utils.openrouter_config.generate_google_content_with_retry')
    def test_gemini_model_success(self, mock_generate_gemini, mock_client_manager):
        """测试Gemini模型成功调用的情况"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {
            "gemini": (self.mock_gemini_client, "gemini-1.5-flash")
        }
        mock_generate_gemini.return_value = self.mock_response_gemini
        
        # 执行测试
        result = get_chat_completion(self.messages, model="gemini")
        
        # 验证结果
        self.assertEqual(result, {"gemini": "Gemini的回复"})
        mock_generate_gemini.assert_called_once()

    @patch('src.utils.openrouter_config.client_manager')
    def test_no_available_clients(self, mock_client_manager):
        """测试没有可用客户端的情况"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {}
        
        # 执行测试
        result = get_chat_completion(self.messages)
        
        # 验证结果
        self.assertEqual(result, {})

    @patch('src.utils.openrouter_config.client_manager')
    def test_client_initialization_failure(self, mock_client_manager):
        """测试客户端初始化失败的情况"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {
            "gemini": (None, "gemini-1.5-flash")
        }
        
        # 执行测试
        result = get_chat_completion(self.messages, model="gemini")
        
        # 验证结果
        self.assertEqual(result, {})

    @patch('src.utils.openrouter_config.client_manager')
    @patch('src.utils.openrouter_config.generate_openai_content_with_retry')
    def test_api_call_retry(self, mock_generate_openai, mock_client_manager):
        """测试API调用重试机制"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {
            "moonshot": (self.mock_openai_client, "moonshot-v1-8k")
        }
        # 前两次调用抛出异常，第三次成功
        mock_generate_openai.side_effect = [
            Exception("API错误"),
            Exception("API错误"),
            self.mock_response_openai
        ]
        
        # 执行测试
        result = get_chat_completion(self.messages, model="moonshot", max_retries=3)
        
        # 验证结果
        self.assertEqual(result, {"moonshot": "OpenAI的回复"})
        self.assertEqual(mock_generate_openai.call_count, 3)

    @patch('src.utils.openrouter_config.client_manager')
    @patch('src.utils.openrouter_config.generate_openai_content_with_retry')
    def test_multiple_models(self, mock_generate_openai, mock_client_manager):
        """测试多个模型同时调用的情况"""
        # 设置模拟返回值
        mock_client_manager.get_clients_info.return_value = {
            "moonshot-1": (self.mock_openai_client, "moonshot-v1-8k"),
            "moonshot-2": (self.mock_openai_client, "moonshot-v1-8k")
        }
        mock_generate_openai.return_value = self.mock_response_openai
        
        # 执行测试
        result = get_chat_completion(self.messages, model=["moonshot-1", "moonshot-2"])
        
        # 验证结果
        self.assertEqual(result, {
            "moonshot-1": "OpenAI的回复",
            "moonshot-2": "OpenAI的回复"
        })
        self.assertEqual(mock_generate_openai.call_count, 2)

if __name__ == '__main__':
    unittest.main()
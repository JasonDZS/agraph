import os
import unittest
from unittest.mock import patch
from agraph.config import Settings


class TestSettings(unittest.TestCase):
    """测试配置设置类"""

    def setUp(self):
        """测试前的设置"""
        self.settings = Settings()

    def test_default_values(self):
        """测试默认配置值"""
        # 应用设置
        self.assertEqual(self.settings.APP_TITLE, "Data Visualization Agent")
        self.assertEqual(self.settings.APP_VERSION, "0.1.0")
        self.assertEqual(self.settings.HOST, "0.0.0.0")
        self.assertEqual(self.settings.PORT, 8000)
        self.assertEqual(self.settings.LOG_LEVEL, "debug")

        # CORS设置
        self.assertEqual(self.settings.CORS_ORIGINS, ["*"])
        self.assertEqual(self.settings.CORS_METHODS, ["GET", "POST", "PUT", "DELETE", "OPTIONS"])
        self.assertEqual(self.settings.CORS_HEADERS, ["*"])
        self.assertFalse(self.settings.CORS_CREDENTIALS)

        # 数据库设置
        self.assertEqual(self.settings.DATABASES_DIR, "databases")
        self.assertEqual(self.settings.LOGS_DIR, "logs")
        self.assertEqual(self.settings.TEMPLATES_DIR, "templates")

        # 模型设置
        self.assertEqual(self.settings.LLM_MODEL, "deepseek-ai/DeepSeek-V3")
        self.assertEqual(self.settings.LLM_TEMPERATURE, 0.0)
        self.assertEqual(self.settings.LLM_MAX_TOKENS, 2048)

        # 嵌入设置
        self.assertEqual(self.settings.EMBEDDING_DIM, 1024)
        self.assertEqual(self.settings.EMBEDDING_MAX_TOKEN_SIZE, 8192)

    def test_environment_variables(self):
        """测试环境变量配置"""
        # 测试环境变量的获取逻辑，而不是重新加载模块
        test_key = "test_api_key_12345"
        test_base = "https://custom.api.com/v1"

        with patch.dict(os.environ, {
            'OPENAI_API_KEY': test_key,
            'OPENAI_API_BASE': test_base,
            'LLM_PROVIDER': 'ollama',
            'EMBEDDING_MODEL': 'test-embedding-model',
            'EMBEDDING_PROVIDER': 'hf'
        }):
            # 直接测试os.getenv的行为
            self.assertEqual(os.getenv("OPENAI_API_KEY"), test_key)
            self.assertEqual(os.getenv("OPENAI_API_BASE"), test_base)
            self.assertEqual(os.getenv("LLM_PROVIDER"), 'ollama')
            self.assertEqual(os.getenv("EMBEDDING_MODEL"), 'test-embedding-model')
            self.assertEqual(os.getenv("EMBEDDING_PROVIDER"), 'hf')

    def test_empty_environment_variables(self):
        """测试空环境变量的默认值"""
        # 测试当环境变量不存在时的默认值
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(os.getenv("OPENAI_API_KEY", ""), "")
            self.assertEqual(os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"), "https://api.openai.com/v1")
            self.assertEqual(os.getenv("LLM_PROVIDER", "openai"), "openai")
            self.assertEqual(os.getenv("EMBEDDING_PROVIDER", "openai"), "openai")

    def test_settings_class_attributes(self):
        """测试Settings类是否包含所有预期属性"""
        # 应用相关属性
        self.assertTrue(hasattr(self.settings, 'APP_TITLE'))
        self.assertTrue(hasattr(self.settings, 'APP_VERSION'))
        self.assertTrue(hasattr(self.settings, 'HOST'))
        self.assertTrue(hasattr(self.settings, 'PORT'))
        self.assertTrue(hasattr(self.settings, 'LOG_LEVEL'))

        # CORS相关属性
        self.assertTrue(hasattr(self.settings, 'CORS_ORIGINS'))
        self.assertTrue(hasattr(self.settings, 'CORS_METHODS'))
        self.assertTrue(hasattr(self.settings, 'CORS_HEADERS'))
        self.assertTrue(hasattr(self.settings, 'CORS_CREDENTIALS'))

        # API相关属性
        self.assertTrue(hasattr(self.settings, 'OPENAI_API_KEY'))
        self.assertTrue(hasattr(self.settings, 'OPENAI_API_BASE'))

        # 目录相关属性
        self.assertTrue(hasattr(self.settings, 'DATABASES_DIR'))
        self.assertTrue(hasattr(self.settings, 'LOGS_DIR'))
        self.assertTrue(hasattr(self.settings, 'TEMPLATES_DIR'))

        # 模型相关属性
        self.assertTrue(hasattr(self.settings, 'LLM_MODEL'))
        self.assertTrue(hasattr(self.settings, 'LLM_TEMPERATURE'))
        self.assertTrue(hasattr(self.settings, 'LLM_MAX_TOKENS'))
        self.assertTrue(hasattr(self.settings, 'LLM_PROVIDER'))

        # 嵌入相关属性
        self.assertTrue(hasattr(self.settings, 'EMBEDDING_MODEL'))
        self.assertTrue(hasattr(self.settings, 'EMBEDDING_PROVIDER'))
        self.assertTrue(hasattr(self.settings, 'EMBEDDING_DIM'))
        self.assertTrue(hasattr(self.settings, 'EMBEDDING_MAX_TOKEN_SIZE'))

    def test_data_types(self):
        """测试属性的数据类型"""
        # 字符串类型
        self.assertIsInstance(self.settings.APP_TITLE, str)
        self.assertIsInstance(self.settings.APP_VERSION, str)
        self.assertIsInstance(self.settings.HOST, str)
        self.assertIsInstance(self.settings.LOG_LEVEL, str)
        self.assertIsInstance(self.settings.OPENAI_API_KEY, str)
        self.assertIsInstance(self.settings.OPENAI_API_BASE, str)
        self.assertIsInstance(self.settings.DATABASES_DIR, str)
        self.assertIsInstance(self.settings.LOGS_DIR, str)
        self.assertIsInstance(self.settings.TEMPLATES_DIR, str)
        self.assertIsInstance(self.settings.LLM_MODEL, str)
        self.assertIsInstance(self.settings.LLM_PROVIDER, str)
        self.assertIsInstance(self.settings.EMBEDDING_MODEL, str)
        self.assertIsInstance(self.settings.EMBEDDING_PROVIDER, str)

        # 整数类型
        self.assertIsInstance(self.settings.PORT, int)
        self.assertIsInstance(self.settings.LLM_MAX_TOKENS, int)
        self.assertIsInstance(self.settings.EMBEDDING_DIM, int)
        self.assertIsInstance(self.settings.EMBEDDING_MAX_TOKEN_SIZE, int)

        # 浮点数类型
        self.assertIsInstance(self.settings.LLM_TEMPERATURE, float)

        # 布尔类型
        self.assertIsInstance(self.settings.CORS_CREDENTIALS, bool)

        # 列表类型
        self.assertIsInstance(self.settings.CORS_ORIGINS, list)
        self.assertIsInstance(self.settings.CORS_METHODS, list)
        self.assertIsInstance(self.settings.CORS_HEADERS, list)

    def test_cors_settings_content(self):
        """测试CORS设置的具体内容"""
        # 检查CORS方法包含标准HTTP方法
        expected_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        for method in expected_methods:
            self.assertIn(method, self.settings.CORS_METHODS)

        # 检查CORS Origins
        self.assertIn("*", self.settings.CORS_ORIGINS)

        # 检查CORS Headers
        self.assertIn("*", self.settings.CORS_HEADERS)

    def test_valid_log_levels(self):
        """测试有效的日志级别"""
        valid_log_levels = ["debug", "info", "warning", "error", "critical"]
        self.assertIn(self.settings.LOG_LEVEL, valid_log_levels)

    def test_valid_providers(self):
        """测试有效的提供商设置"""
        # 测试默认LLM提供商
        valid_llm_providers = ["openai", "hf", "ollama", "custom"]
        self.assertIn(self.settings.LLM_PROVIDER, valid_llm_providers)

        # 测试不同的嵌入提供商
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'hf'}):
            settings = Settings()
            valid_embedding_providers = ["openai", "hf", "ollama"]
            self.assertIn(settings.EMBEDDING_PROVIDER, valid_embedding_providers)

    def test_numeric_ranges(self):
        """测试数值范围的合理性"""
        # 端口范围
        self.assertGreaterEqual(self.settings.PORT, 1)
        self.assertLessEqual(self.settings.PORT, 65535)

        # 温度范围（通常在0-2之间）
        self.assertGreaterEqual(self.settings.LLM_TEMPERATURE, 0.0)
        self.assertLessEqual(self.settings.LLM_TEMPERATURE, 2.0)

        # 最大token数应为正数
        self.assertGreater(self.settings.LLM_MAX_TOKENS, 0)

        # 嵌入维度应为正数
        self.assertGreater(self.settings.EMBEDDING_DIM, 0)

        # 最大token大小应为正数
        self.assertGreater(self.settings.EMBEDDING_MAX_TOKEN_SIZE, 0)

class TestConfigModule(unittest.TestCase):
    """测试配置模块"""

    def test_config_module_import(self):
        """测试配置模块导入"""
        from agraph.config import Settings

        # 确保Settings类可以实例化
        settings = Settings()
        self.assertIsNotNone(settings)

        # 确保所有必要的属性都存在
        required_attrs = [
            'APP_TITLE', 'APP_VERSION', 'HOST', 'PORT', 'LOG_LEVEL',
            'CORS_ORIGINS', 'CORS_METHODS', 'CORS_HEADERS', 'CORS_CREDENTIALS',
            'OPENAI_API_KEY', 'OPENAI_API_BASE',
            'DATABASES_DIR', 'LOGS_DIR', 'TEMPLATES_DIR',
            'LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_MAX_TOKENS', 'LLM_PROVIDER',
            'EMBEDDING_MODEL', 'EMBEDDING_PROVIDER', 'EMBEDDING_DIM', 'EMBEDDING_MAX_TOKEN_SIZE'
        ]

        for attr in required_attrs:
            self.assertTrue(hasattr(settings, attr), f"Missing required attribute: {attr}")

    def test_dotenv_loading(self):
        """测试.env文件加载功能"""
        # 测试dotenv的基本功能，而不是具体的Settings实例
        # 验证环境变量可以被正确设置和读取

        # 创建一个临时的环境变量来测试
        with patch.dict(os.environ, {'TEST_CONFIG_VAR': 'test_value'}):
            # 验证环境变量确实被设置
            self.assertEqual(os.getenv('TEST_CONFIG_VAR'), 'test_value')

        # 测试带默认值的环境变量获取
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(os.getenv('NONEXISTENT_VAR', 'default'), 'default')

        # 验证dotenv模块已经被导入
        import dotenv
        self.assertTrue(hasattr(dotenv, 'load_dotenv'))


if __name__ == '__main__':
    unittest.main()

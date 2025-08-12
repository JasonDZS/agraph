"""
Test cases for the config module.
"""

import os
import unittest
from unittest.mock import patch

from agraph.config import Settings


class TestSettings(unittest.TestCase):
    """Test cases for the Settings class."""

    def test_settings_defaults(self):
        """Test that settings have reasonable defaults."""
        settings = Settings()

        self.assertEqual(settings.workdir, "workdir")
        self.assertEqual(settings.llm.temperature, 0.0)
        self.assertIsInstance(settings.llm.max_tokens, int)
        self.assertEqual(settings.llm.provider, "openai")

    @patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key-123", "LLM_MODEL": "gpt-4", "LLM_MAX_TOKENS": "8192"},
    )
    def test_settings_from_environment(self):
        """Test that settings can be loaded from environment variables."""
        settings = Settings()

        self.assertEqual(settings.openai.api_key, "test-key-123")
        self.assertEqual(settings.llm.model, "gpt-4")
        self.assertEqual(settings.llm.max_tokens, 8192)

    def test_settings_custom_values(self):
        """Test creating settings with custom values."""
        from agraph.config import LLMConfig

        settings = Settings(
            workdir="/custom/workdir", llm=LLMConfig(model="custom-model", temperature=0.5)
        )

        self.assertEqual(settings.workdir, "/custom/workdir")
        self.assertEqual(settings.llm.model, "custom-model")
        self.assertEqual(settings.llm.temperature, 0.5)

    def test_settings_validation(self):
        """Test that settings validation works."""
        from agraph.config import LLMConfig

        # Test valid temperature
        llm_config = LLMConfig(temperature=0.5)
        self.assertEqual(llm_config.temperature, 0.5)

        # Test that extreme temperatures can be created (no built-in validation)
        llm_config_low = LLMConfig(temperature=-1.0)
        self.assertEqual(llm_config_low.temperature, -1.0)


if __name__ == "__main__":
    unittest.main()

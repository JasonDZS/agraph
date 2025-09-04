"""Tests for Chat API router."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from agraph.api.app import app
from agraph.api.models import ResponseStatus


class TestChatRouter(unittest.TestCase):
    """Test Chat API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_success(self, mock_get_agraph):
        """Test successful non-streaming chat."""
        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_chat_response = {
            "answer": "This is the answer to your question",
            "context": {
                "entities": [{"id": "e1", "name": "Entity 1"}],
                "relations": [{"id": "r1", "type": "KNOWS"}],
                "text_chunks": [{"id": "t1", "content": "Relevant text"}],
            },
            "confidence": 0.85,
        }
        mock_agraph.chat.return_value = mock_chat_response
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "question": "What is the main topic?",
            "conversation_history": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
            "entity_top_k": 5,
            "relation_top_k": 5,
            "text_chunk_top_k": 5,
            "response_type": "详细回答",
            "stream": False,
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)
        self.assertEqual(data["message"], "Chat response generated successfully")
        self.assertEqual(data["data"], mock_chat_response)

        # Verify agraph.chat was called with correct parameters
        mock_agraph.chat.assert_called_once_with(
            question="What is the main topic?",
            conversation_history=test_data["conversation_history"],
            entity_top_k=5,
            relation_top_k=5,
            text_chunk_top_k=5,
            response_type="详细回答",
            stream=False,
        )

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_with_project_name(self, mock_get_agraph):
        """Test chat with specific project name."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Project-specific answer"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "What is this project about?", "stream": False}

        response = self.client.post("/chat?project_name=test_project", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify get_agraph_instance was called with project name
        mock_get_agraph.assert_called_once_with("test_project")

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_stream_flag_error(self, mock_get_agraph):
        """Test chat endpoint rejects streaming requests."""
        mock_agraph = AsyncMock()
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question", "stream": True}  # Should trigger error

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Use /chat/stream endpoint", data["message"])

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_invalid_response_format(self, mock_get_agraph):
        """Test chat with invalid response format from agraph."""
        # Mock AGraph that returns non-dict response
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = "Invalid response format"  # Should be dict
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question", "stream": False}

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Invalid response format", data["message"])

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_agraph_error(self, mock_get_agraph):
        """Test chat when AGraph raises exception."""
        # Mock AGraph that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.chat.side_effect = Exception("Chat processing failed")
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question", "stream": False}

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Chat processing failed")

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_minimal_request(self, mock_get_agraph):
        """Test chat with minimal request data."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Minimal response"}
        mock_get_agraph.return_value = mock_agraph

        # Only provide required fields
        test_data = {"question": "Simple question"}

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)

        # Verify default values are used
        mock_agraph.chat.assert_called_once()
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertEqual(call_kwargs["entity_top_k"], 5)  # Default value
        self.assertEqual(call_kwargs["relation_top_k"], 5)  # Default value
        self.assertEqual(call_kwargs["text_chunk_top_k"], 5)  # Default value
        self.assertEqual(call_kwargs["response_type"], "详细回答")  # Default value
        self.assertFalse(call_kwargs["stream"])

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_custom_parameters(self, mock_get_agraph):
        """Test chat with custom parameters."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Custom response"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "question": "Custom question",
            "entity_top_k": 10,
            "relation_top_k": 8,
            "text_chunk_top_k": 15,
            "response_type": "简洁回答",
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify custom parameters were passed
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertEqual(call_kwargs["entity_top_k"], 10)
        self.assertEqual(call_kwargs["relation_top_k"], 8)
        self.assertEqual(call_kwargs["text_chunk_top_k"], 15)
        self.assertEqual(call_kwargs["response_type"], "简洁回答")


class TestChatStreamingRouter(unittest.TestCase):
    """Test Chat Streaming API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_stream_success(self, mock_get_agraph):
        """Test successful streaming chat."""

        # Mock streaming response generator
        async def mock_chat_generator():
            yield {
                "question": "Test question",
                "chunk": "First",
                "partial_answer": "First",
                "finished": False,
            }
            yield {
                "question": "Test question",
                "chunk": " chunk",
                "partial_answer": "First chunk",
                "finished": False,
            }
            yield {
                "question": "Test question",
                "chunk": "",
                "partial_answer": "First chunk",
                "answer": "First chunk",
                "finished": True,
                "context": {"entities": [], "relations": []},
            }

        # Mock AGraph instance
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = mock_chat_generator()
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question", "stream": True}

        response = self.client.post("/chat/stream", json=test_data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")
        self.assertIn("no-cache", response.headers.get("cache-control", ""))

        # Verify AGraph chat was called with stream=True
        mock_agraph.chat.assert_called_once()
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertTrue(call_kwargs["stream"])

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_stream_non_streaming_response(self, mock_get_agraph):
        """Test streaming endpoint with non-streaming response."""
        # Mock AGraph that returns dict instead of generator
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {
            "question": "Test question",
            "answer": "Non-streaming answer",
            "finished": True,
        }
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question"}

        response = self.client.post("/chat/stream", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Should handle non-streaming response gracefully
        # Response should still be in streaming format

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_stream_agraph_error(self, mock_get_agraph):
        """Test streaming chat when AGraph raises exception."""
        # Mock AGraph that raises exception
        mock_agraph = AsyncMock()
        mock_agraph.chat.side_effect = Exception("Streaming chat failed")
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Test question"}

        response = self.client.post("/chat/stream", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Streaming chat failed")

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_stream_with_project_name(self, mock_get_agraph):
        """Test streaming chat with specific project name."""

        # Mock streaming response
        async def mock_generator():
            yield {
                "question": "Project question",
                "chunk": "Project",
                "partial_answer": "Project",
                "finished": False,
            }

        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = mock_generator()
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "Project question"}

        response = self.client.post("/chat/stream?project_name=my_project", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify get_agraph_instance was called with project name
        mock_get_agraph.assert_called_once_with("my_project")

    def test_chat_validation_missing_question(self):
        """Test chat with missing required question field."""
        test_data = {}  # Missing question

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    def test_chat_validation_invalid_types(self):
        """Test chat with invalid parameter types."""
        test_data = {
            "question": "Test question",
            "entity_top_k": "invalid",  # Should be int
            "stream": "invalid",  # Should be bool
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 422)  # Pydantic validation error

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_with_conversation_history(self, mock_get_agraph):
        """Test chat with conversation history."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Response with context"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "question": "Follow-up question",
            "conversation_history": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence"},
                {"role": "user", "content": "How does it work?"},
                {"role": "assistant", "content": "It uses algorithms and data"},
            ],
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify conversation history was passed correctly
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertEqual(call_kwargs["conversation_history"], test_data["conversation_history"])

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_edge_case_empty_conversation_history(self, mock_get_agraph):
        """Test chat with empty conversation history."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "First response"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": "First question", "conversation_history": []}  # Empty history

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Should handle empty history gracefully
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertEqual(call_kwargs["conversation_history"], [])

    def test_chat_stream_response_format(self):
        """Test that streaming response has correct format."""
        with patch("agraph.api.routers.chat.get_agraph_instance") as mock_get_agraph:
            # Mock simple streaming response
            async def simple_generator():
                yield {
                    "question": "Test",
                    "chunk": "Hello",
                    "partial_answer": "Hello",
                    "finished": True,
                }

            mock_agraph = AsyncMock()
            mock_agraph.chat.return_value = simple_generator()
            mock_get_agraph.return_value = mock_agraph

            test_data = {"question": "Test question"}

            response = self.client.post("/chat/stream", json=test_data)

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")

            # Verify streaming headers
            self.assertIn("no-cache", response.headers.get("cache-control", ""))

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_instance_acquisition_error(self, mock_get_agraph):
        """Test chat when AGraph instance acquisition fails."""
        mock_get_agraph.side_effect = Exception("Failed to get AGraph instance")

        test_data = {"question": "Test question"}

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "Failed to get AGraph instance")

    def test_chat_request_model_defaults(self):
        """Test that ChatRequest model uses correct defaults."""
        # This tests the Pydantic model defaults indirectly
        with patch("agraph.api.routers.chat.get_agraph_instance") as mock_get_agraph:
            mock_agraph = AsyncMock()
            mock_agraph.chat.return_value = {"answer": "Default test"}
            mock_get_agraph.return_value = mock_agraph

            # Minimal request to test defaults
            test_data = {"question": "Test with defaults"}

            response = self.client.post("/chat", json=test_data)

            self.assertEqual(response.status_code, 200)

            # Verify defaults were applied
            call_kwargs = mock_agraph.chat.call_args.kwargs
            self.assertEqual(call_kwargs["entity_top_k"], 5)
            self.assertEqual(call_kwargs["relation_top_k"], 5)
            self.assertEqual(call_kwargs["text_chunk_top_k"], 5)
            self.assertEqual(call_kwargs["response_type"], "详细回答")
            self.assertFalse(call_kwargs["stream"])
            self.assertIsNone(call_kwargs["conversation_history"])

    def test_chat_question_length_limits(self):
        """Test chat with very long question."""
        with patch("agraph.api.routers.chat.get_agraph_instance") as mock_get_agraph:
            mock_agraph = AsyncMock()
            mock_agraph.chat.return_value = {"answer": "Long question response"}
            mock_get_agraph.return_value = mock_agraph

            # Very long question
            long_question = "What is AI? " * 1000  # Very long question
            test_data = {"question": long_question}

            response = self.client.post("/chat", json=test_data)

            # Should handle long questions (API level doesn't restrict length)
            self.assertEqual(response.status_code, 200)

            # Verify the full question was passed
            call_kwargs = mock_agraph.chat.call_args.kwargs
            self.assertEqual(call_kwargs["question"], long_question)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_top_k_parameters(self, mock_get_agraph):
        """Test chat with various top_k parameter values."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Top-k response"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {
            "question": "Test question",
            "entity_top_k": 20,
            "relation_top_k": 15,
            "text_chunk_top_k": 25,
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)

        # Verify custom top_k values were passed
        call_kwargs = mock_agraph.chat.call_args.kwargs
        self.assertEqual(call_kwargs["entity_top_k"], 20)
        self.assertEqual(call_kwargs["relation_top_k"], 15)
        self.assertEqual(call_kwargs["text_chunk_top_k"], 25)

    def test_chat_response_type_variations(self):
        """Test chat with different response type values."""
        with patch("agraph.api.routers.chat.get_agraph_instance") as mock_get_agraph:
            mock_agraph = AsyncMock()
            mock_agraph.chat.return_value = {"answer": "Brief response"}
            mock_get_agraph.return_value = mock_agraph

            test_cases = ["简洁回答", "详细回答", "分析报告", "custom_type"]

            for response_type in test_cases:
                test_data = {
                    "question": f"Question for {response_type}",
                    "response_type": response_type,
                }

                response = self.client.post("/chat", json=test_data)

                self.assertEqual(response.status_code, 200)

                # Verify response type was passed correctly
                call_kwargs = mock_agraph.chat.call_args.kwargs
                self.assertEqual(call_kwargs["response_type"], response_type)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_empty_question(self, mock_get_agraph):
        """Test chat with empty question string."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Empty question response"}
        mock_get_agraph.return_value = mock_agraph

        test_data = {"question": ""}  # Empty question

        response = self.client.post("/chat", json=test_data)

        # API should accept empty question and let AGraph handle it
        self.assertEqual(response.status_code, 200)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_instance_not_initialized(self, mock_get_agraph):
        """Test chat when AGraph instance is not properly initialized."""
        # This would typically be handled by the dependency injection
        # but we can test the error propagation
        mock_get_agraph.side_effect = Exception("AGraph not initialized")

        test_data = {"question": "Test question"}

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["message"], "AGraph not initialized")


class TestChatIntegration(unittest.TestCase):
    """Integration tests for chat functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_chat_endpoints_exist(self):
        """Test that chat endpoints are properly registered."""
        # Test non-streaming endpoint
        response = self.client.post("/chat", json={"question": "test"})
        # Should not return 404 (endpoint exists)
        self.assertNotEqual(response.status_code, 404)

        # Test streaming endpoint
        response = self.client.post("/chat/stream", json={"question": "test"})
        # Should not return 404 (endpoint exists)
        self.assertNotEqual(response.status_code, 404)

    @patch("agraph.api.routers.chat.get_agraph_instance")
    def test_chat_request_validation_comprehensive(self, mock_get_agraph):
        """Test comprehensive request validation."""
        mock_agraph = AsyncMock()
        mock_agraph.chat.return_value = {"answer": "Valid response"}
        mock_get_agraph.return_value = mock_agraph

        # Valid request with all optional fields
        test_data = {
            "question": "Comprehensive test question",
            "conversation_history": [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
            "entity_top_k": 7,
            "relation_top_k": 6,
            "text_chunk_top_k": 8,
            "response_type": "分析报告",
            "stream": False,
        }

        response = self.client.post("/chat", json=test_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], ResponseStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()

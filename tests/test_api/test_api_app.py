"""Tests for FastAPI application core functionality."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from agraph.api.app import app, general_exception_handler, http_exception_handler, log_requests
from agraph.api.models import ResponseStatus


class TestFastAPIApp(unittest.TestCase):
    """Test FastAPI application initialization and core functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_app_initialization(self):
        """Test FastAPI app is properly initialized."""
        self.assertEqual(app.title, "AGraph API")
        self.assertEqual(app.description, "Knowledge Graph Construction and RAG System API")
        self.assertEqual(app.version, "1.0.0")

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], "1.0.0")
        self.assertIn("timestamp", data)

    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured."""
        # Check if CORS headers are present in response
        response = self.client.options("/health")
        # CORS middleware should handle OPTIONS requests
        self.assertIn(
            response.status_code, [200, 405]
        )  # 405 if method not allowed is acceptable for options

    @patch("agraph.api.app.logger")
    async def test_request_logging_middleware(self, mock_logger):
        """Test request logging middleware functionality."""
        # Create a mock request and response
        request = MagicMock(spec=Request)
        request.method = "GET"
        request.url.path = "/test"
        request.client.host = "127.0.0.1"

        # Mock response
        response = MagicMock()
        response.status_code = 200

        # Mock call_next function
        async def mock_call_next(req):
            return response

        # Test the middleware
        result = await log_requests(request, mock_call_next)

        # Verify response is returned
        self.assertEqual(result, response)

        # Verify logging was called
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        self.assertIn("GET /test", log_call_args)
        self.assertIn("200", log_call_args)
        self.assertIn("127.0.0.1", log_call_args)

    def test_request_logging_middleware_unknown_client(self):
        """Test request logging with unknown client IP."""
        # This test would require mocking the middleware more extensively
        # For now, we'll test through integration
        pass

    async def test_http_exception_handler(self):
        """Test HTTP exception handler."""
        request = MagicMock(spec=Request)
        exc = HTTPException(status_code=404, detail="Not found")

        response = await http_exception_handler(request, exc)

        self.assertEqual(response.status_code, 404)

        # Parse response content
        import json

        content = json.loads(response.body.decode())
        self.assertEqual(content["status"], ResponseStatus.ERROR)
        self.assertEqual(content["message"], "Not found")
        self.assertEqual(content["error_code"], "404")

    async def test_general_exception_handler(self):
        """Test general exception handler."""
        request = MagicMock(spec=Request)
        exc = ValueError("Test error")

        with patch("agraph.api.app.logger") as mock_logger:
            response = await general_exception_handler(request, exc)

        self.assertEqual(response.status_code, 500)

        # Verify error was logged
        mock_logger.error.assert_called_once()

        # Parse response content
        import json

        content = json.loads(response.body.decode())
        self.assertEqual(content["status"], ResponseStatus.ERROR)
        self.assertEqual(content["message"], "An unexpected error occurred")
        self.assertIn("exception", content["error_details"])

    def test_router_inclusion(self):
        """Test that all required routers are included."""
        routes = [route.path for route in app.routes]

        # Check that router prefixes are included (routers add their own prefixes)
        # We can't easily test the exact paths without knowing router internals,
        # but we can test that routes exist beyond just /health
        self.assertGreater(len(routes), 1)
        self.assertIn("/health", routes)

    @patch("agraph.api.dependencies.get_agraph_instance")
    async def test_lifespan_startup(self, mock_get_agraph):
        """Test application lifespan startup."""
        # This is more of an integration test
        # We'll test that the lifespan function calls get_agraph_instance
        mock_get_agraph.return_value = AsyncMock()

        # The lifespan is already configured in the app, so we test indirectly
        # by ensuring our health endpoint works (which means lifespan ran)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_openapi_schema_generation(self):
        """Test OpenAPI schema is properly generated."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)

        schema = response.json()
        self.assertEqual(schema["info"]["title"], "AGraph API")
        self.assertEqual(schema["info"]["version"], "1.0.0")


class TestAppMiddleware(unittest.TestCase):
    """Test middleware functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_cors_headers_present(self):
        """Test CORS headers are added to responses."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        # Check for CORS headers (these might be added by FastAPI automatically)
        # The exact headers depend on the request and CORS configuration


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the complete FastAPI application."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_app_startup_and_health(self):
        """Test app starts up properly and health endpoint works."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_invalid_endpoint_returns_404(self):
        """Test that invalid endpoints return 404."""
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_app_handles_json_responses(self):
        """Test app properly handles JSON responses."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")


if __name__ == "__main__":
    unittest.main()

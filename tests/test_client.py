"""Tests for pando.client — PandoClient subprocess mode."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pando.client import PandoClient
from pando.exceptions import PandoConnectionError, PandoTimeoutError
from pando.types import RunResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    """Create a mock asyncio subprocess."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPandoClientInit:
    def test_defaults(self):
        client = PandoClient()
        assert client._cwd is None
        assert client._model is None
        assert client._timeout == 300.0

    def test_custom_params(self):
        client = PandoClient(cwd="/tmp", model="claude-sonnet-4-6", timeout=60)
        assert client._cwd == "/tmp"
        assert client._model == "claude-sonnet-4-6"
        assert client._timeout == 60.0


class TestPandoClientBuildCmd:
    def test_basic_command(self):
        with patch("pando.transport._find_pando_binary", return_value="/usr/bin/pando"):
            client = PandoClient()
            client._pando_path = "/usr/bin/pando"
            cmd = client._build_cmd("hello world")
        assert cmd == ["/usr/bin/pando", "-p", "hello world", "-f", "json"]

    def test_with_model(self):
        client = PandoClient(model="copilot.gpt-5.4")
        client._pando_path = "/usr/bin/pando"
        cmd = client._build_cmd("test")
        assert "-m" in cmd
        assert "copilot.gpt-5.4" in cmd

    def test_with_cwd(self):
        client = PandoClient(cwd="/my/project")
        client._pando_path = "/usr/bin/pando"
        cmd = client._build_cmd("test")
        assert "-c" in cmd
        assert "/my/project" in cmd

    def test_with_yolo(self):
        client = PandoClient()
        client._pando_path = "/usr/bin/pando"
        cmd = client._build_cmd("test", allow_all_tools=True)
        assert "--yolo" in cmd

    def test_without_yolo(self):
        client = PandoClient()
        client._pando_path = "/usr/bin/pando"
        cmd = client._build_cmd("test", allow_all_tools=False)
        assert "--yolo" not in cmd


class TestParseOutput:
    def test_valid_json_response(self):
        output = json.dumps({"response": "All done.", "sessionId": "abc123"})
        result = PandoClient._parse_output(output)
        assert result.response == "All done."
        assert result.session_id == "abc123"
        assert result.raw["response"] == "All done."

    def test_json_on_last_line(self):
        # Simulate pando emitting log lines before the JSON result
        output = "INFO Starting pando...\n" + json.dumps({"response": "Done!"})
        result = PandoClient._parse_output(output)
        assert result.response == "Done!"

    def test_fallback_plain_text(self):
        output = "some plain text output"
        result = PandoClient._parse_output(output)
        assert result.response == "some plain text output"
        assert result.session_id == ""

    def test_empty_output(self):
        result = PandoClient._parse_output("")
        assert result.response == ""

    def test_json_without_response_key(self):
        # JSON that doesn't have 'response' key falls back to plain text.
        output = json.dumps({"error": "something failed"})
        result = PandoClient._parse_output(output)
        # The fallback returns the raw stdout as plain text
        assert "error" in result.response or result.response == output.strip()


class TestPandoClientRunAsync:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        response_data = {"response": "Task completed successfully."}
        process = _make_process(stdout=json.dumps(response_data).encode())

        with (
            patch("pando.transport._find_pando_binary", return_value="/usr/bin/pando"),
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            client = PandoClient()
            result = await client.run_async("Fix lint errors")

        assert isinstance(result, RunResult)
        assert result.response == "Task completed successfully."

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_connection_error(self):
        process = _make_process(
            stdout=b"", stderr=b"something failed", returncode=1
        )

        with (
            patch("pando.transport._find_pando_binary", return_value="/usr/bin/pando"),
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
        ):
            client = PandoClient()
            with pytest.raises(PandoConnectionError, match="exited with code 1"):
                await client.run_async("test")

    @pytest.mark.asyncio
    async def test_timeout_raises_pando_timeout_error(self):
        import asyncio

        process = _make_process()
        process.communicate = AsyncMock(side_effect=asyncio.TimeoutError)

        with (
            patch("pando.transport._find_pando_binary", return_value="/usr/bin/pando"),
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("asyncio.wait_for", side_effect=asyncio.TimeoutError),
        ):
            client = PandoClient(timeout=1.0)
            with pytest.raises(PandoTimeoutError):
                await client.run_async("test")

    @pytest.mark.asyncio
    async def test_os_error_raises_connection_error(self):
        with (
            patch("pando.transport._find_pando_binary", return_value="/usr/bin/pando"),
            patch(
                "asyncio.create_subprocess_exec",
                AsyncMock(side_effect=OSError("No such file")),
            ),
        ):
            client = PandoClient()
            with pytest.raises(PandoConnectionError, match="Failed to start pando"):
                await client.run_async("test")

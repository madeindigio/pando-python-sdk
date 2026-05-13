"""PandoClient — simple non-interactive subprocess mode.

This module provides :class:`PandoClient` which runs ``pando -p "..."`` as a
subprocess and returns a :class:`~pando.types.RunResult` with the response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from asyncio import subprocess as aiosubprocess

from pando.exceptions import (
    PandoBinaryNotFoundError,
    PandoConnectionError,
    PandoTimeoutError,
)
from pando.transport import _find_pando_binary
from pando.types import RunResult

logger = logging.getLogger(__name__)


class PandoClient:
    """Non-interactive subprocess client for pando.

    Runs ``pando -p "<prompt>" -f json`` and returns the result. This mode is
    the simplest way to invoke pando for one-shot tasks without maintaining a
    long-running connection.

    Example::

        from pando import PandoClient

        client = PandoClient(cwd="/path/to/project", model="copilot.gpt-5.4")
        result = client.run("Fix all lint errors", allow_all_tools=True)
        print(result.response)

    Args:
        cwd: Working directory for the pando subprocess. Defaults to the
            current working directory.
        model: Optional model identifier (e.g. ``"copilot.gpt-5.4"``).
        pando_path: Explicit path to the pando binary.
        timeout: Maximum seconds to wait for the subprocess to complete.
            Defaults to 300 (5 minutes).
    """

    def __init__(
        self,
        cwd: str | None = None,
        model: str | None = None,
        pando_path: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self._cwd = cwd
        self._model = model
        self._pando_path: str | None = None
        self._raw_pando_path = pando_path
        self._timeout = timeout

    def _get_binary(self) -> str:
        if self._pando_path is None:
            self._pando_path = _find_pando_binary(self._raw_pando_path)
        return self._pando_path

    def _build_cmd(self, prompt: str, allow_all_tools: bool = False) -> list[str]:
        """Build the subprocess command line arguments."""
        binary = self._get_binary()
        cmd = [binary, "-p", prompt, "-f", "json"]
        if self._model:
            cmd += ["-m", self._model]
        if self._cwd:
            cmd += ["-c", self._cwd]
        if allow_all_tools:
            cmd.append("--yolo")
        return cmd

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def run_async(
        self,
        prompt: str,
        allow_all_tools: bool = False,
    ) -> RunResult:
        """Async variant of :meth:`run`.

        Args:
            prompt: The user prompt to pass to pando.
            allow_all_tools: If ``True``, passes ``--yolo`` to auto-approve all
                tool permission requests.

        Returns:
            :class:`~pando.types.RunResult` with the assistant response.

        Raises:
            :class:`~pando.exceptions.PandoBinaryNotFoundError`: If pando cannot
                be located.
            :class:`~pando.exceptions.PandoConnectionError`: If the subprocess
                fails to start or exits with a non-zero code.
            :class:`~pando.exceptions.PandoTimeoutError`: If the subprocess
                exceeds :attr:`timeout` seconds.
        """
        cmd = self._build_cmd(prompt, allow_all_tools)
        logger.debug("Running pando command: %s", cmd)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=aiosubprocess.PIPE,
                stderr=aiosubprocess.PIPE,
                cwd=self._cwd,
            )
        except (OSError, FileNotFoundError) as exc:
            raise PandoConnectionError(f"Failed to start pando: {exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise PandoTimeoutError(
                f"pando did not complete within {self._timeout:.0f} seconds"
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            detail = stderr.strip() or stdout.strip()
            raise PandoConnectionError(
                f"pando exited with code {process.returncode}: {detail[:500]}"
            )

        return self._parse_output(stdout)

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        allow_all_tools: bool = False,
    ) -> RunResult:
        """Run a single prompt and return the result (synchronous).

        Creates a new event loop internally to run the async implementation.

        Example::

            client = PandoClient(cwd="/my/project")
            result = client.run("Explain the auth module")
            print(result.response)

        Args:
            prompt: The user prompt.
            allow_all_tools: Auto-approve all tool permissions (``--yolo`` flag).

        Returns:
            :class:`~pando.types.RunResult` with the assistant response.

        Raises:
            :class:`~pando.exceptions.PandoBinaryNotFoundError`: If pando is not found.
            :class:`~pando.exceptions.PandoConnectionError`: If the subprocess fails.
            :class:`~pando.exceptions.PandoTimeoutError`: If timeout is exceeded.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running inside an existing event loop — use run_coroutine_threadsafe.
                import concurrent.futures
                future: concurrent.futures.Future = asyncio.run_coroutine_threadsafe(
                    self.run_async(prompt, allow_all_tools), loop
                )
                return future.result(timeout=self._timeout + 10)
        except RuntimeError:
            pass

        return asyncio.run(self.run_async(prompt, allow_all_tools))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_output(stdout: str) -> RunResult:
        """Parse the pando JSON output into a :class:`~pando.types.RunResult`."""
        # pando -f json may emit log lines before the final JSON object.
        # Scan backwards to find the last complete JSON line.
        lines = stdout.strip().splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "response" in data:
                    return RunResult(
                        response=data["response"],
                        session_id=data.get("sessionId") or data.get("session_id", ""),
                        raw=data,
                    )
            except json.JSONDecodeError:
                continue

        # Fallback: return the entire stdout as plain text.
        return RunResult(response=stdout.strip())

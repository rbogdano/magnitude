"""A stand-in for an EIM serving container.

Speaks the parts of the surface ICN depends on -- /health, /v1/models, and a streaming
/v1/chat/completions -- so the container lifecycle, readiness polling, and chat transport can be
exercised without vLLM. That matters because vLLM's CPU backend needs AVX-512, which a developer
machine generally lacks, and because a real model load takes minutes even when it works.

It is deliberately unfaithful in one way: it reports a served model name unrelated to
INFERENCE_MODEL_ID. vLLM validates requests against its own served name, and EIM's test helper
carries a warning comment about that mismatch, so the stub makes any code that assumes otherwise
fail loudly.

Behavior is driven by the same environment EIM reads, plus STUB_* knobs:

  STUB_SLOW_START     seconds of 503 from /health before reporting ready
  STUB_CRASH_AFTER    exit with code 17 this many seconds after start
  STUB_FAIL_MESSAGE   print this to stderr and exit 1 immediately, to exercise log
                      classification against EIM's real failure strings
  STUB_REASONING      emit reasoning_content deltas before the answer
  STUB_TOOL_CALL      emit an incremental tool call instead of prose
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

STARTED = time.time()
SLOW_START = float(os.environ.get("STUB_SLOW_START", "0"))
CRASH_AFTER = float(os.environ.get("STUB_CRASH_AFTER", "0"))
FAIL_MESSAGE = os.environ.get("STUB_FAIL_MESSAGE", "")
REASONING = os.environ.get("STUB_REASONING", "") not in ("", "0")
TOOL_CALL = os.environ.get("STUB_TOOL_CALL", "") not in ("", "0")

# Proves the served name is read rather than assumed.
SERVED_MODEL_NAME = "stub/" + os.environ.get("INFERENCE_MODEL_ID", "unknown").split("/")[-1]

if FAIL_MESSAGE:
    # EIM exits 0 or 1 and nothing else, so the only signal is the log text.
    print(FAIL_MESSAGE, file=sys.stderr, flush=True)
    raise SystemExit(1)

if CRASH_AFTER:
    threading.Timer(CRASH_AFTER, lambda: os._exit(17)).start()


def sse(payload):
    return f"data: {json.dumps(payload)}\n\n".encode()


def chunk(delta, finish_reason=None):
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "model": SERVED_MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _empty(self, status):
        self.send_response(status)
        self.send_header("content-length", "0")
        self.end_headers()

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            # Ready is 200 with an empty body, exactly as vLLM reports it.
            self._empty(200 if time.time() - STARTED >= SLOW_START else 503)
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "stub"}]})
        else:
            self._empty(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._empty(404)
            return

        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")

        if request.get("model") != SERVED_MODEL_NAME:
            # The same rejection vLLM gives, so the caller must read /v1/models.
            body = json.dumps({"error": {
                "message": f"model {request.get('model')!r} is not served",
                "type": "NotFoundError"}}).encode()
            self.send_response(404)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        self.wfile.write(sse(chunk({"role": "assistant"})))
        if REASONING:
            for text in ("I should ", "answer briefly."):
                self.wfile.write(sse(chunk({"reasoning_content": text})))

        if TOOL_CALL:
            # Streamed in fragments, the way vLLM emits arguments.
            self.wfile.write(sse(chunk({"tool_calls": [{
                "index": 0, "id": "call_stub", "type": "function",
                "function": {"name": "read_file", "arguments": '{"pa'}}]})))
            self.wfile.write(sse(chunk({"tool_calls": [{
                "index": 0, "function": {"arguments": 'th":"a.rs"}'}}]})))
            finish_reason = "tool_calls"
        else:
            for text in ("Hello", " from", " the", " stub."):
                self.wfile.write(sse(chunk({"content": text})))
            finish_reason = "stop"

        self.wfile.write(sse(chunk({}, finish_reason)))
        # Usage arrives in its own final chunk when stream_options.include_usage was requested.
        if request.get("stream_options", {}).get("include_usage"):
            usage = {
                "id": "chatcmpl-stub", "object": "chat.completion.chunk",
                "model": SERVED_MODEL_NAME, "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15,
                          "prompt_tokens_details": {"cached_tokens": 8}},
            }
            self.wfile.write(sse(usage))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    # Report the launch contract so a failing test can show what the container was given.
    print(json.dumps({
        "servedModelName": SERVED_MODEL_NAME,
        "inferenceModelId": os.environ.get("INFERENCE_MODEL_ID"),
        "inferenceProfileId": os.environ.get("INFERENCE_PROFILE_ID"),
        "inferenceEngineArgs": os.environ.get("INFERENCE_ENGINE_ARGS"),
    }), flush=True)
    HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()

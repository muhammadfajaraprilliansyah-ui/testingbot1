"""Agent keputusan: ClaudeAgent (tool use) dan RandomAgent (baseline)."""

import json
import logging
import random
import time
from typing import Dict, List, Optional

logger = logging.getLogger("idxagent.agent")


class AgentFatalError(Exception):
    """Error fatal agent yang tidak dapat dipulihkan (mis. API Key salah, model tidak ditemukan)."""
    pass

SYSTEM_PROMPT = """You are an autonomous trading agent inside a HISTORICAL simulation of
the Indonesian stock market (IDX). A Python simulator gives you a scale-free
market snapshot at the close of each trading day.

Execution rules (enforced by the simulator, you cannot bypass them):
- Your decisions are made at today's close and filled at TOMORROW's open.
- stop_loss_pct and take_profit_pct are percentages measured from the ACTUAL fill price.
- Stops are checked intraday; gaps can fill you worse than your stop.
- Round-trip trading cost is given in the rules payload. Small edges can be eaten by costs.
- Position size is capped by max_position_pct and by the per-trade risk budget.
- The simulator validates and may reject orders; rejected orders are not retried.

Symbols and dates are anonymised. Do NOT try to identify the real stocks or
dates, and do not use any knowledge of what happened later. Decide only from
the supplied data, your portfolio and your own past experience.

You choose your own methodology. Do not follow any single indicator blindly.
Doing nothing (empty decisions list) is a valid and often correct decision.
Never BUY and EXIT the same symbol in one response.

Always answer by calling the submit_decisions tool."""

TOOL = {
    "name": "submit_decisions",
    "description": "Submit today's trading decisions for the simulator to validate and execute.",
    "input_schema": {
        "type": "object",
        "properties": {
            "market_view": {"type": "string", "description": "Short view of current conditions."},
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["BUY", "EXIT"]},
                        "symbol": {"type": "string"},
                        "size_pct_equity": {"type": "number", "description": "Target position size, % of equity (BUY only)."},
                        "stop_loss_pct": {"type": "number", "description": "Stop distance below fill, % (BUY only)."},
                        "take_profit_pct": {"type": "number", "description": "Target distance above fill, % (BUY only)."},
                        "reason": {"type": "string"},
                    },
                    "required": ["action", "symbol", "reason"],
                },
            },
            "learning_note": {"type": "string", "description": "What you learned from recent experience (may be empty)."},
        },
        "required": ["decisions", "market_view", "learning_note"],
    },
}


class ClaudeAgent:
    def __init__(self, api_key: str, model: str, max_retries: int = 5, client=None):
        if client is not None:
            self.client = client
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key, max_retries=2)
        self.model = model
        self.max_retries = max_retries
        self.experience: List[Dict] = []
        self.notes: List[str] = []
        self.last_view = ""
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_tokens": 0, "cache_write_tokens": 0}

    # ------------------------------------------------------------ memory
    def add_experience(self, summary: Dict) -> None:
        self.experience = (self.experience + [summary])[-30:]

    def memory_summary(self) -> Dict:
        if not self.experience:
            return {"closed_trades": 0, "recent_trades": [], "your_previous_learning_notes": self.notes[-5:]}
        rets = [e["return_pct"] for e in self.experience]
        return {
            "closed_trades": len(self.experience),
            "recent_win_rate_pct": round(sum(r > 0 for r in rets) / len(rets) * 100, 1),
            "recent_avg_return_pct": round(sum(rets) / len(rets), 2),
            "recent_trades": self.experience[-10:],
            "your_previous_learning_notes": self.notes[-5:],
        }

    # ---------------------------------------------------------- decision
    def decide(self, ctx: Dict) -> Dict:
        import anthropic

        payload = {**ctx, "agent_memory": self.memory_summary()}
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=[{"type": "text", "text": SYSTEM_PROMPT,
                             "cache_control": {"type": "ephemeral"}}],
                    tools=[TOOL],
                    tool_choice={"type": "tool", "name": TOOL["name"]},
                    messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                )
                u = resp.usage
                self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                self.usage["cache_read_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
                self.usage["cache_write_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0

                text_snippets = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use" and block.name == TOOL["name"]:
                        out = dict(block.input)
                        out.setdefault("decisions", [])
                        note = (out.get("learning_note") or "").strip()
                        if note:
                            self.notes = (self.notes + [note])[-10:]
                        self.last_view = out.get("market_view", "")
                        return out
                    if getattr(block, "type", None) == "text" and block.text:
                        text_snippets.append(block.text.strip())

                snippet_msg = f" Respon model: {' '.join(text_snippets)[:200]}" if text_snippets else ""
                raise ValueError(f"Claude tidak memanggil tool submit_decisions.{snippet_msg}")

            except anthropic.AuthenticationError as e:
                raise AgentFatalError(f"Autentikasi Anthropic gagal (API Key tidak valid): {e}") from e

            except anthropic.NotFoundError as e:
                raise AgentFatalError(f"Model Anthropic '{self.model}' tidak ditemukan (404): {e}") from e

            except anthropic.PermissionDeniedError as e:
                raise AgentFatalError(f"Akses ditolak ke model Anthropic '{self.model}' (403): {e}") from e

            except anthropic.BadRequestError as e:
                raise ValueError(f"Request Anthropic tidak valid (BadRequestError): {e}") from e

            except anthropic.RateLimitError as e:
                last_error = e
                # Ambil delay dari header Retry-After jika tersedia
                retry_after = getattr(e, "response", None)
                wait_sec = None
                if retry_after is not None:
                    hdr = getattr(retry_after, "headers", {}).get("retry-after")
                    if hdr:
                        try:
                            wait_sec = float(hdr)
                        except (ValueError, TypeError):
                            pass
                if wait_sec is None:
                    wait_sec = min(60.0, (2.0 ** attempt) + random.uniform(0.5, 1.5))
                logger.warning(
                    "Rate limit Anthropic tercapai (429). Menunggu %.1f detik (percobaan %d/%d)...",
                    wait_sec, attempt, self.max_retries
                )
                if attempt < self.max_retries:
                    time.sleep(wait_sec)
                    continue
                raise ValueError(f"Rate limit Anthropic terlampaui setelah {self.max_retries} percobaan: {e}") from e

            except anthropic.APIConnectionError as e:
                last_error = e
                wait_sec = min(30.0, (2.0 ** attempt) + random.uniform(0.5, 1.5))
                logger.warning(
                    "Gangguan koneksi API Anthropic (%s). Menunggu %.1f detik (percobaan %d/%d)...",
                    e, wait_sec, attempt, self.max_retries
                )
                if attempt < self.max_retries:
                    time.sleep(wait_sec)
                    continue
                raise ValueError(f"Koneksi API Anthropic terputus setelah {self.max_retries} percobaan: {e}") from e

            except anthropic.APIStatusError as e:
                last_error = e
                # 529 = Overloaded, 500+ = internal server error
                if e.status_code in (429, 500, 502, 503, 504, 529):
                    wait_sec = min(60.0, (2.0 ** attempt) + random.uniform(1.0, 2.5))
                    logger.warning(
                        "Server Anthropic sibuk (status %d). Menunggu %.1f detik (percobaan %d/%d)...",
                        e.status_code, wait_sec, attempt, self.max_retries
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait_sec)
                        continue
                raise ValueError(f"Error status API Anthropic ({e.status_code}): {e}") from e

        if last_error:
            raise ValueError(f"Panggilan Claude gagal setelah {self.max_retries} percobaan: {last_error}") from last_error
        raise ValueError("Panggilan Claude gagal.")


class RandomAgent:
    """Baseline: beli simbol acak dengan SL/TP berbasis ATR, aturan risiko sama."""

    def __init__(self, seed: int = 0, p_buy: float = 0.15):
        self.rng = random.Random(seed)
        self.p_buy = p_buy
        self.usage = {}

    def add_experience(self, summary: Dict) -> None:
        pass

    def decide(self, ctx: Dict) -> Dict:
        held = {p["symbol"] for p in ctx["portfolio"]["open_positions"]}
        cands = [s for s in ctx["market"] if s not in held]
        decisions = []
        if ctx["rules"]["open_slots"] > 0 and cands and self.rng.random() < self.p_buy:
            s = self.rng.choice(cands)
            atr = ctx["market"][s]["atr_pct"]
            decisions.append({
                "action": "BUY", "symbol": s, "size_pct_equity": 20,
                "stop_loss_pct": round(max(1.0, 1.5 * atr), 2),
                "take_profit_pct": round(max(1.5, 3.0 * atr), 2),
                "reason": "random baseline",
            })
        return {"decisions": decisions, "market_view": "", "learning_note": ""}

"""
Quick wiring test for the Groq LLM integration.

Verifies the exact path used by app.py:
    .streamlit/secrets.toml  ->  generate_explanation()  ->  Groq API

It instruments the live Groq HTTP call so we can tell the difference between:
    1. A real LLM explanation was fetched from Groq           (exit 0)
    2. The groove call FAILED and the deterministic fallback  (exit 2)
       was silently used instead (the app shows template text,
       not AI-generated text)
    3. Wiring broken (no key, import error, etc.)              (exit 1)

Usage:
    .venv/bin/python scripts/test_groq_wiring.py
"""

import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_explanation import DEFAULT_MODEL, GroqLLMClient, parse_explanation


def _read_secret(name: str) -> Optional[str]:
    """Read a value out of .streamlit/secrets.toml (same file st.secrets reads)."""
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None
    match = re.search(
        rf'^\s*{name}\s*=\s*"([^"]+)"',
        secrets_path.read_text(),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def sample_detections() -> list:
    """Mimic the structured output produced by run_inspection()/calculate_severities()."""
    return [
        {
            "class": "Crack",
            "confidence": 0.87,
            "bbox": [120, 80, 160, 110],
            "location": "middle body",
            "severity": "High",
            "area_ratio": 0.04,
        }
    ]


def main() -> int:
    api_key = _read_secret("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .streamlit/secrets.toml")
        return 1

    runtime_model = _read_secret("GROQ_MODEL") or DEFAULT_MODEL
    print(
        f"✅ [1/4] API key loaded from .streamlit/secrets.toml ({api_key[:8]}...)"
        f" | runtime model: '{runtime_model}'"
    )

    # --- Verify the key + model against Groq directly (proves SDK + key) ---
    client = GroqLLMClient(api_key=api_key, model=runtime_model)
    try:
        page = client._get_client().models.list()
        available = {m.id for m in page.data}
        print(f"✅ [2/4] Groq SDK connected. {len(available)} models on this account.")
    except Exception as exc:
        print(f"❌ [2/4] Groq SDK connection failed: {exc}")
        return 1

    if runtime_model not in available:
        print(
            f"❌ [2/4] Model '{runtime_model}' is NOT available on this Groq account.\n"
            f"   Available chat models: "
            f"{sorted(sorted(available), key=str)[:20]}\n"
            f"   Fix GROQ_MODEL in .streamlit/secrets.toml before testing again."
        )
        return 1

    # --- Instrument the live HTTP call so we KNOW if Groq was hit ---
    # generate_explanation() builds its own GroqLLMClient(), so we patch the
    # class-level _get_client to make every instance reuse one instrumented
    # underlying Groq() client we can spy on.
    state = {"attempts": 0, "error": None}
    import llm_explanation as llm_mod

    real_client = client._get_client()  # the actual Groq() instance with our key
    original_create = real_client.chat.completions.create

    def spying_create(*args, **kwargs):
        state["attempts"] += 1
        try:
            return original_create(*args, **kwargs)
        except Exception as exc:  # the SDK will bubble this up to generate()
            state["error"] = repr(exc)
            raise

    real_client.chat.completions.create = spying_create
    llm_mod.GroqLLMClient._get_client = lambda self: real_client

    print(f"🔍 [3/4] Calling generate_explanation() ...\n")
    from llm_explanation import generate_explanation

    try:
        text = generate_explanation(sample_detections(), api_key=api_key, model=runtime_model)
    except Exception as exc:
        print(f"❌ [3/4] generate_explanation() raised: {exc}")
        return 1

    if state["attempts"] == 0:
        print("⚠️  [3/4] No Groq call attempted - deterministic FALLBACK used.")
        return 2
    if state["error"]:
        print(
            f"⚠️  [3/4] Groq call FAILED and deterministic FALLBACK was used.\n"
            f"        Error: {state['error']}"
        )
        return 2

    print(f"✅ [3/4] Live Groq call succeeded (real AI-generated explanation).\n")
    print("=" * 60)
    print("GROQ RESPONSE")
    print("=" * 60)
    print(text)
    print("=" * 60)

    parsed = parse_explanation(text)
    print("\n✅ [4/4] parse_explanation() extracted:")
    print(f"   - Description:      {parsed['description'][:80]}")
    print(f"   - Possible causes:  {len(parsed['possible_causes'])} listed")
    print(f"   - Impact:           {parsed['impact'][:60]}")
    print(f"   - Recommended act:  {parsed['recommended_action'][:60]}")

    print(
        f"\n🎉 Wiring is correct! app.py -> generate_explanation() -> "
        f"Groq('{runtime_model}') works end-to-end."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
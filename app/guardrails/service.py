import yaml
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logfire

from langchain_groq import ChatGroq
from nemoguardrails import LLMRails, RailsConfig
from config import settings


class GuardrailsService:
    """
    Service managing NeMo Guardrails lifecycle, custom actions,
    and query safety classification.
    """

    def __init__(self, prompts_path: Optional[Path] = None):
        self._prompts_path = prompts_path or Path(__file__).parent / "prompts.yml"
        self._rails: Optional[LLMRails] = None
        self._safety_llm: Optional[ChatGroq] = None
        self._prompts: Dict[str, Any] = {}
        self.load_prompts()

    def load_prompts(self) -> None:
        """Loads prompt templates and refusal responses from YAML configuration."""
        try:
            if self._prompts_path.exists():
                with open(self._prompts_path, "r", encoding="utf-8") as f:
                    self._prompts = yaml.safe_load(f) or {}
                logfire.debug(f"Loaded guardrails prompts from {self._prompts_path}")
            else:
                logfire.error(f"Prompts file not found at {self._prompts_path}")
        except Exception as exc:
            logfire.error(f"Failed to load guardrails prompts: {type(exc).__name__}: {exc}")

    @property
    def safety_prompt_template(self) -> str:
        return self._prompts.get("safety_classifier", {}).get("system_prompt", "")

    @property
    def jailbreak_refusal(self) -> str:
        return self._prompts.get("refusals", {}).get(
            "jailbreak",
            "Your request has been blocked due to a detected security violation."
        ).strip()

    @property
    def off_topic_refusal(self) -> str:
        return self._prompts.get("refusals", {}).get(
            "off_topic",
            "Your request has been blocked as it is outside the scope of allowed topics."
        ).strip()

    def initialize(self) -> None:
        """
        Called once at application lifespan startup.
        Initializes LLMRails config, Groq LLM instance, and registers custom action.
        """
        if not settings.GROQ_API_KEY:
            logfire.error("GROQ_API_KEY is not set. NeMo Guardrails cannot initialize.")
            return

        self._safety_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0,
            max_tokens=10,
        )

        config = RailsConfig.from_path(settings.NEMOGUARDRAILS_CONFIG_PATH)
        self._rails = LLMRails(config, llm=self._safety_llm)
        self._rails.register_action(self.check_input_safety, name="check_input_safety")

        logfire.info("NeMo Guardrails service successfully initialized.")
        logfire.info(f"  LLM    : Groq / {settings.GROQ_MODEL}")
        logfire.info(f"  Config : {settings.NEMOGUARDRAILS_CONFIG_PATH}")
        logfire.info(f"  Action : check_input_safety registered")

    def get_status(self) -> dict:
        """Returns current guardrails operational status."""
        return {
            "initialized": self._rails is not None,
            "llm": f"groq/{settings.GROQ_MODEL}" if self._rails else None,
            "config_path": settings.NEMOGUARDRAILS_CONFIG_PATH,
            "prompts_loaded": bool(self._prompts),
        }

    async def check_input_safety(self, context: Optional[dict] = None) -> str:
        """
        NeMo Guardrails custom action — registered via rails.register_action().

        Makes one Groq LLM call to classify the user message for both
        jailbreak detection and off-topic filtering in a single pass.

        Args:
            context: NeMo runtime context containing "user_message".

        Returns:
            "jailbreak" | "off_topic" | "pass"
        """
        if self._safety_llm is None:
            logfire.warning("Safety LLM not initialized.")
            return "off_topic"

        user_message = (
            context.get("user_message", "") if context else ""
        ).strip()

        if not user_message:
            return "pass"

        template = self.safety_prompt_template
        if not template:
            logfire.error("Safety prompt template missing.")
            return "off_topic"

        prompt = template.replace("{user_message}", user_message)

        try:
            with logfire.span("Groq Safety Classification", query=user_message[:80]):
                response = await self._safety_llm.ainvoke(prompt)
                result = response.content.strip().lower()

            logfire.info(f"Safety result: '{result}' | query: '{user_message[:60]}'")

            if "jailbreak" in result:
                logfire.warning(f"JAILBREAK: '{user_message[:80]}'")
                return "jailbreak"

            if "off_topic" in result or "off-topic" in result:
                logfire.warning(f"OFF-TOPIC: '{user_message[:80]}'")
                return "off_topic"

            return "pass"

        except Exception as exc:
            logfire.error(f"Safety classification error: {type(exc).__name__}: {exc}")
            return "off_topic"

    async def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Runs the input safety validation against the user query.

        Returns:
            (True, None)         — safe and on-topic
            (False, refusal_msg) — blocked with refusal message
        """
        if not query or not query.strip():
            return True, None

        if self._rails is None:
            logfire.error("NeMo Guardrails not initialized.")
            return False, "The safety system is not ready. Please try again in a moment."

        with logfire.span("NeMo Guardrails Gate", query=query[:80]):
            try:
                context = {"user_message": query}
                result = await self.check_input_safety(context=context)

                if result == "jailbreak":
                    logfire.warning(f"BLOCKED (jailbreak): '{query[:60]}'")
                    return False, self.jailbreak_refusal

                if result == "off_topic":
                    logfire.warning(f"BLOCKED (off-topic): '{query[:60]}'")
                    return False, self.off_topic_refusal

                logfire.info(f"PASSED: '{query[:60]}'")
                return True, None

            except Exception as exc:
                logfire.error(f"Guardrails error: {type(exc).__name__}: {exc}")
                return False, "The safety system encountered an error. Please try again."

guardrails_service = GuardrailsService()

def initialize_guardrails() -> None:
    guardrails_service.initialize()


def get_guardrails() -> dict:
    return guardrails_service.get_status()


async def check_input_safety(context: Optional[dict] = None) -> str:
    return await guardrails_service.check_input_safety(context=context)


async def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    return await guardrails_service.validate_query(query=query)

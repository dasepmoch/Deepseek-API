"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from .auth import Session, get_session, login
from .client import DeepSeekClient, Reply
from .pow import DeepSeekPow

__version__ = "0.1.0"

__all__ = ["Session", "get_session", "login", "DeepSeekClient", "Reply", "DeepSeekPow"]
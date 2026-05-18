"""Authentication module — JWT-based token validation."""

import hashlib
import hmac
import time
from dataclasses import dataclass


SECRET_KEY = "change-me-in-production"
TOKEN_EXPIRY = 3600  # 1 hour


@dataclass
class User:
    id: int
    email: str
    role: str


def create_token(user: User) -> str:
    """Create a signed JWT token for the given user."""
    payload = f"{user.id}:{user.email}:{user.role}:{int(time.time()) + TOKEN_EXPIRY}"
    signature = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def validate_token(token: str) -> User | None:
    """Validate a JWT token and return the User, or None if invalid/expired."""
    try:
        payload, signature = token.rsplit(".", 1)
        expected = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        parts = payload.split(":")
        if int(parts[3]) < int(time.time()):
            return None  # expired
        return User(id=int(parts[0]), email=parts[1], role=parts[2])
    except (ValueError, IndexError):
        return None


class AuthMiddleware:
    """ASGI middleware that validates JWT tokens on incoming requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode()
        if auth.startswith("Bearer "):
            user = validate_token(auth[7:])
            scope["user"] = user
        return await self.app(scope, receive, send)

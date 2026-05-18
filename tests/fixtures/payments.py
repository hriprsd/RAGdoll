"""Payment processing — Stripe-style charge + refund handling."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Payment:
    id: str
    amount_cents: int
    currency: str
    status: PaymentStatus
    customer_id: str
    description: Optional[str] = None


class PaymentProcessor:
    """Process payments via external gateway (Stripe-like interface)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._ledger: dict[str, Payment] = {}

    def charge(self, customer_id: str, amount_cents: int, currency: str = "usd") -> Payment:
        """Create a new charge. Returns the Payment record."""
        import hashlib, time
        payment_id = hashlib.sha256(f"{customer_id}{time.time()}".encode()).hexdigest()[:12]
        payment = Payment(
            id=payment_id,
            amount_cents=amount_cents,
            currency=currency,
            status=PaymentStatus.COMPLETED,
            customer_id=customer_id,
        )
        self._ledger[payment_id] = payment
        return payment

    def refund(self, payment_id: str) -> Payment:
        """Refund a completed payment. Raises ValueError if not refundable."""
        payment = self._ledger.get(payment_id)
        if not payment:
            raise ValueError(f"Payment not found: {payment_id}")
        if payment.status != PaymentStatus.COMPLETED:
            raise ValueError(f"Cannot refund payment in state: {payment.status.value}")
        payment.status = PaymentStatus.REFUNDED
        return payment

    def get_payment(self, payment_id: str) -> Optional[Payment]:
        return self._ledger.get(payment_id)

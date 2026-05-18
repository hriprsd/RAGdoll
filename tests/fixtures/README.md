# My Service

A backend API for handling payments, authentication, and rate limiting.

## Architecture

The service uses JWT-based authentication. Tokens are validated on every request
via the AuthMiddleware. We chose JWT over sessions because the mobile client
cannot handle cookies.

## Database

SQLite with WAL mode and a simple connection pool. Pagination is handled via
LIMIT/OFFSET with a total count query.

## Rate Limiting

Sliding window counter per client IP. Clients exceeding 200 requests/minute
are temporarily banned. See `rate_limiter.py` for implementation.

## Payments

Stripe-like charge and refund flow. The PaymentProcessor maintains an in-memory
ledger for development; production uses the Stripe API.

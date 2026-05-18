interface PaginationParams {
  offset: number;
  limit: number;
}

interface PaginatedResponse<T> {
  items: T[];
  total: number;
  hasMore: boolean;
}

type PaymentStatus = "pending" | "completed" | "failed" | "refunded";

interface Payment {
  id: string;
  amountCents: number;
  currency: string;
  status: PaymentStatus;
  customerId: string;
}

export async function handleListPayments(
  params: PaginationParams
): Promise<PaginatedResponse<Payment>> {
  const { offset, limit } = params;
  // In production this queries the payments table
  const payments: Payment[] = [];
  return {
    items: payments.slice(offset, offset + limit),
    total: payments.length,
    hasMore: offset + limit < payments.length,
  };
}

export async function handleCreateCharge(
  customerId: string,
  amountCents: number,
  currency: string = "usd"
): Promise<Payment> {
  const payment: Payment = {
    id: crypto.randomUUID(),
    amountCents,
    currency,
    status: "completed",
    customerId,
  };
  return payment;
}

export async function handleRefund(paymentId: string): Promise<Payment> {
  // Look up the payment and mark as refunded
  throw new Error(`Payment not found: ${paymentId}`);
}

export function handleHealthCheck(): { status: string; timestamp: string } {
  return {
    status: "ok",
    timestamp: new Date().toISOString(),
  };
}

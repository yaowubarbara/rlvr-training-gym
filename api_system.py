"""
API Training Gym - Module 1: System Clone
A simulated order management API with state machine transitions.
The LLM agent must learn to call APIs in the correct sequence.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from enum import Enum
import uuid
import random

app = FastAPI(title="Order Management API - Training Gym")

# ===== State Machine =====
class OrderStatus(str, Enum):
    CREATED = "created"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

# Valid state transitions
VALID_TRANSITIONS = {
    OrderStatus.CREATED: [OrderStatus.PAID, OrderStatus.CANCELLED],
    OrderStatus.PAID: [OrderStatus.SHIPPED, OrderStatus.REFUNDED, OrderStatus.CANCELLED],
    OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
    OrderStatus.DELIVERED: [OrderStatus.REFUNDED],
    OrderStatus.CANCELLED: [],
    OrderStatus.REFUNDED: [],
}

# ===== Models =====
class OrderCreate(BaseModel):
    customer_name: str
    product: str
    quantity: int
    price: float

class PaymentRequest(BaseModel):
    order_id: str
    amount: float
    method: str  # "card" or "bank_transfer"

class ShipmentRequest(BaseModel):
    order_id: str
    address: str
    carrier: str  # "dhl", "fedex", "ups"

class RefundRequest(BaseModel):
    order_id: str
    reason: str

class Order(BaseModel):
    id: str
    customer_name: str
    product: str
    quantity: int
    price: float
    total: float
    status: OrderStatus
    payment_method: Optional[str] = None
    shipping_address: Optional[str] = None
    carrier: Optional[str] = None
    refund_reason: Optional[str] = None

# ===== In-memory store =====
orders: dict[str, Order] = {}

# ===== API Endpoints =====

@app.post("/orders", response_model=Order)
def create_order(req: OrderCreate):
    """Create a new order. Returns order with status 'created'."""
    if req.quantity <= 0:
        raise HTTPException(400, "Quantity must be positive")
    if req.price <= 0:
        raise HTTPException(400, "Price must be positive")

    order_id = str(uuid.uuid4())[:8]
    order = Order(
        id=order_id,
        customer_name=req.customer_name,
        product=req.product,
        quantity=req.quantity,
        price=req.price,
        total=round(req.quantity * req.price, 2),
        status=OrderStatus.CREATED,
    )
    orders[order_id] = order
    return order

@app.get("/orders/{order_id}", response_model=Order)
def get_order(order_id: str):
    """Get order details by ID."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")
    return orders[order_id]

@app.get("/orders", response_model=list[Order])
def list_orders():
    """List all orders."""
    return list(orders.values())

@app.post("/orders/{order_id}/pay", response_model=Order)
def pay_order(order_id: str, req: PaymentRequest):
    """Pay for an order. Order must be in 'created' status."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")

    order = orders[order_id]

    if req.order_id != order_id:
        raise HTTPException(400, "Order ID mismatch in request body")

    if order.status != OrderStatus.CREATED:
        raise HTTPException(400, f"Cannot pay order in '{order.status}' status. Must be 'created'.")

    if req.method not in ("card", "bank_transfer"):
        raise HTTPException(400, "Payment method must be 'card' or 'bank_transfer'")

    if abs(req.amount - order.total) > 0.01:
        raise HTTPException(400, f"Payment amount {req.amount} doesn't match order total {order.total}")

    order.status = OrderStatus.PAID
    order.payment_method = req.method
    orders[order_id] = order
    return order

@app.post("/orders/{order_id}/ship", response_model=Order)
def ship_order(order_id: str, req: ShipmentRequest):
    """Ship an order. Order must be in 'paid' status."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")

    order = orders[order_id]

    if req.order_id != order_id:
        raise HTTPException(400, "Order ID mismatch in request body")

    if order.status != OrderStatus.PAID:
        raise HTTPException(400, f"Cannot ship order in '{order.status}' status. Must be 'paid'.")

    if req.carrier not in ("dhl", "fedex", "ups"):
        raise HTTPException(400, "Carrier must be 'dhl', 'fedex', or 'ups'")

    if not req.address.strip():
        raise HTTPException(400, "Address cannot be empty")

    order.status = OrderStatus.SHIPPED
    order.shipping_address = req.address
    order.carrier = req.carrier
    orders[order_id] = order
    return order

@app.post("/orders/{order_id}/deliver", response_model=Order)
def deliver_order(order_id: str):
    """Mark order as delivered. Order must be in 'shipped' status."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")

    order = orders[order_id]

    if order.status != OrderStatus.SHIPPED:
        raise HTTPException(400, f"Cannot deliver order in '{order.status}' status. Must be 'shipped'.")

    order.status = OrderStatus.DELIVERED
    orders[order_id] = order
    return order

@app.post("/orders/{order_id}/cancel", response_model=Order)
def cancel_order(order_id: str):
    """Cancel an order. Only 'created' or 'paid' orders can be cancelled."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")

    order = orders[order_id]

    if order.status not in (OrderStatus.CREATED, OrderStatus.PAID):
        raise HTTPException(400, f"Cannot cancel order in '{order.status}' status.")

    order.status = OrderStatus.CANCELLED
    orders[order_id] = order
    return order

@app.post("/orders/{order_id}/refund", response_model=Order)
def refund_order(order_id: str, req: RefundRequest):
    """Refund an order. Only 'paid' or 'delivered' orders can be refunded."""
    if order_id not in orders:
        raise HTTPException(404, f"Order {order_id} not found")

    order = orders[order_id]

    if req.order_id != order_id:
        raise HTTPException(400, "Order ID mismatch in request body")

    if order.status not in (OrderStatus.PAID, OrderStatus.DELIVERED):
        raise HTTPException(400, f"Cannot refund order in '{order.status}' status.")

    if not req.reason.strip():
        raise HTTPException(400, "Refund reason cannot be empty")

    order.status = OrderStatus.REFUNDED
    order.refund_reason = req.reason
    orders[order_id] = order
    return order

@app.post("/reset")
def reset_system():
    """Reset all orders. Used between training episodes."""
    orders.clear()
    return {"status": "reset", "orders_count": 0}


# ===== API Schema (for the LLM agent) =====
API_SCHEMA = """
## Order Management API

### Endpoints:
1. POST /orders — Create order (body: {customer_name, product, quantity, price})
2. GET /orders/{id} — Get order details
3. GET /orders — List all orders
4. POST /orders/{id}/pay — Pay (body: {order_id, amount, method:"card"|"bank_transfer"})
5. POST /orders/{id}/ship — Ship (body: {order_id, address, carrier:"dhl"|"fedex"|"ups"})
6. POST /orders/{id}/deliver — Mark delivered (no body)
7. POST /orders/{id}/cancel — Cancel (no body, only created/paid orders)
8. POST /orders/{id}/refund — Refund (body: {order_id, reason}, only paid/delivered orders)

### State Machine:
created → paid → shipped → delivered → refunded
created → cancelled
paid → cancelled
paid → refunded
delivered → refunded

### Rules:
- Payment amount must exactly match order total (quantity × price)
- Carrier must be one of: dhl, fedex, ups
- Payment method must be: card or bank_transfer
- order_id in request body must match URL parameter
"""

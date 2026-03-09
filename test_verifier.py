"""
Verifier Unit Tests — 10 correct + 10 incorrect agent outputs
Proves the RLVR reward signal is reliable.
Run: python test_verifier.py (requires API server on port 8000)
"""
import json
import subprocess
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_generator import Task
from verifier import execute_and_verify, parse_agent_actions

API_BASE = "http://127.0.0.1:8000"
PASSED = 0
FAILED = 0


def check(name: str, result, expect_success: bool):
    global PASSED, FAILED
    ok = result.success == expect_success
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAILED += 1
        print(f"  [{status}] {name}")
        print(f"         Expected success={expect_success}, got success={result.success}")
        print(f"         Errors: {result.errors}")
        print(f"         Reward: {result.reward}, Partial: {result.partial_reward}")
    else:
        PASSED += 1
        print(f"  [{status}] {name} (reward={result.reward}, partial={result.partial_reward})")


# ============================================================
# 10 CORRECT OUTPUTS — verifier should give reward=1.0
# ============================================================

def test_correct_1_create_order():
    """Single step: create an order."""
    task = Task(
        instruction="Create an order for Alice: 2x Laptop at $999.99",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Alice", "product": "Laptop", "quantity": 2, "price": 999.99}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Alice", "product": "Laptop", "quantity": 2, "price": 999.99}"""
    result = execute_and_verify(task, agent_output)
    check("correct_1_create_order", result, expect_success=True)


def test_correct_2_create_and_check():
    """Two steps: create then GET."""
    task = Task(
        instruction="Create order then check status",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Bob", "product": "Mouse", "quantity": 1, "price": 29.99}},
            {"method": "GET", "path": "/orders/{order_id}"},
        ],
        difficulty=1, task_type="create_and_check"
    )
    # Agent needs to use a placeholder ID — verifier handles dynamic ID
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Bob", "product": "Mouse", "quantity": 1, "price": 29.99}

ACTION: GET /orders/PLACEHOLDER"""
    result = execute_and_verify(task, agent_output)
    # Note: the verifier checks path structure, the PLACEHOLDER will match {order_id}
    check("correct_2_create_and_check", result, expect_success=True)


def test_correct_3_create_and_cancel():
    """Create then cancel immediately."""
    task = Task(
        instruction="Create and cancel",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Claire", "product": "Keyboard", "quantity": 1, "price": 49.99}},
            {"method": "POST", "path": "/orders/{order_id}/cancel"},
        ],
        difficulty=1, task_type="create_and_cancel"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Claire", "product": "Keyboard", "quantity": 1, "price": 49.99}

ACTION: POST /orders/PLACEHOLDER/cancel"""
    result = execute_and_verify(task, agent_output)
    check("correct_3_create_and_cancel", result, expect_success=True)


def test_correct_4_create_and_pay():
    """Create then pay with correct amount."""
    task = Task(
        instruction="Create and pay",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "David", "product": "Monitor", "quantity": 1, "price": 399.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 399.99, "method": "card"}},
        ],
        difficulty=2, task_type="create_and_pay"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "David", "product": "Monitor", "quantity": 1, "price": 399.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 399.99, "method": "card"}"""
    result = execute_and_verify(task, agent_output)
    check("correct_4_create_and_pay", result, expect_success=True)


def test_correct_5_create_pay_ship():
    """Three-step: create, pay, ship."""
    task = Task(
        instruction="Create, pay, ship",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Emma", "product": "SSD 1TB", "quantity": 2, "price": 89.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 179.98, "method": "bank_transfer"}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": "123 Rue de Rivoli, Paris 75001", "carrier": "dhl"}},
        ],
        difficulty=2, task_type="create_pay_ship"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Emma", "product": "SSD 1TB", "quantity": 2, "price": 89.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 179.98, "method": "bank_transfer"}

ACTION: POST /orders/PLACEHOLDER/ship
BODY: {"order_id": "PLACEHOLDER", "address": "123 Rue de Rivoli, Paris 75001", "carrier": "dhl"}"""
    result = execute_and_verify(task, agent_output)
    check("correct_5_create_pay_ship", result, expect_success=True)


def test_correct_6_pay_and_refund():
    """Create, pay, refund."""
    task = Task(
        instruction="Create, pay, refund",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Grace", "product": "Webcam", "quantity": 1, "price": 59.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 59.99, "method": "card"}},
            {"method": "POST", "path": "/orders/{order_id}/refund",
             "body": {"order_id": "{order_id}", "reason": "Product defective"}},
        ],
        difficulty=2, task_type="pay_and_refund"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Grace", "product": "Webcam", "quantity": 1, "price": 59.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 59.99, "method": "card"}

ACTION: POST /orders/PLACEHOLDER/refund
BODY: {"order_id": "PLACEHOLDER", "reason": "Product defective"}"""
    result = execute_and_verify(task, agent_output)
    check("correct_6_pay_and_refund", result, expect_success=True)


def test_correct_7_full_delivery():
    """Four-step: create, pay, ship, deliver."""
    task = Task(
        instruction="Full delivery lifecycle",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Henri", "product": "Tablet", "quantity": 1, "price": 449.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 449.99, "method": "card"}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": "789 Oxford Street, London W1D 1BS", "carrier": "fedex"}},
            {"method": "POST", "path": "/orders/{order_id}/deliver"},
        ],
        difficulty=3, task_type="full_delivery"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Henri", "product": "Tablet", "quantity": 1, "price": 449.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 449.99, "method": "card"}

ACTION: POST /orders/PLACEHOLDER/ship
BODY: {"order_id": "PLACEHOLDER", "address": "789 Oxford Street, London W1D 1BS", "carrier": "fedex"}

ACTION: POST /orders/PLACEHOLDER/deliver"""
    result = execute_and_verify(task, agent_output)
    check("correct_7_full_delivery", result, expect_success=True)


def test_correct_8_full_lifecycle_refund():
    """Five-step: create, pay, ship, deliver, refund."""
    task = Task(
        instruction="Full lifecycle with refund",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Jean", "product": "Headphones", "quantity": 3, "price": 79.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 239.97, "method": "bank_transfer"}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": "321 Friedrichstraße, Berlin 10117", "carrier": "ups"}},
            {"method": "POST", "path": "/orders/{order_id}/deliver"},
            {"method": "POST", "path": "/orders/{order_id}/refund",
             "body": {"order_id": "{order_id}", "reason": "Wrong item received"}},
        ],
        difficulty=3, task_type="full_lifecycle_with_refund"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Jean", "product": "Headphones", "quantity": 3, "price": 79.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 239.97, "method": "bank_transfer"}

ACTION: POST /orders/PLACEHOLDER/ship
BODY: {"order_id": "PLACEHOLDER", "address": "321 Friedrichstraße, Berlin 10117", "carrier": "ups"}

ACTION: POST /orders/PLACEHOLDER/deliver

ACTION: POST /orders/PLACEHOLDER/refund
BODY: {"order_id": "PLACEHOLDER", "reason": "Wrong item received"}"""
    result = execute_and_verify(task, agent_output)
    check("correct_8_full_lifecycle_refund", result, expect_success=True)


def test_correct_9_parse_format():
    """Test parse_agent_actions handles valid format."""
    output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}

ACTION: GET /orders/abc123"""
    actions = parse_agent_actions(output)
    ok = (len(actions) == 2 and
          actions[0]["method"] == "POST" and actions[0]["path"] == "/orders" and
          actions[1]["method"] == "GET" and actions[1]["path"] == "/orders/abc123")
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] correct_9_parse_format (2 actions parsed correctly)")
    else:
        FAILED += 1
        print(f"  [FAIL] correct_9_parse_format — parsed {len(actions)} actions: {actions}")


def test_correct_10_single_item_quantity():
    """Edge case: quantity=1, price has many decimals."""
    task = Task(
        instruction="Create order for 1x USB Cable at $9.99",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Kenji", "product": "USB Cable", "quantity": 1, "price": 9.99}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Kenji", "product": "USB Cable", "quantity": 1, "price": 9.99}"""
    result = execute_and_verify(task, agent_output)
    check("correct_10_single_item", result, expect_success=True)


# ============================================================
# 10 INCORRECT OUTPUTS — verifier should give reward=0.0
# ============================================================

def test_wrong_1_empty_output():
    """Empty output should get reward=0."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}}
        ],
        difficulty=1, task_type="create_order"
    )
    result = execute_and_verify(task, "")
    check("wrong_1_empty_output", result, expect_success=False)


def test_wrong_2_no_action_keyword():
    """Output without ACTION: keyword should fail."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """I'll create an order for you.
POST /orders with body {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_2_no_action_keyword", result, expect_success=False)


def test_wrong_3_wrong_method():
    """GET instead of POST should fail."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: GET /orders
BODY: {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_3_wrong_method", result, expect_success=False)


def test_wrong_4_wrong_path():
    """Wrong path (/order instead of /orders) should fail."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: POST /order
BODY: {"customer_name": "Test", "product": "X", "quantity": 1, "price": 10.0}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_4_wrong_path", result, expect_success=False)


def test_wrong_5_wrong_payment_amount():
    """Payment amount doesn't match total — API should reject."""
    task = Task(
        instruction="Create and pay",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "Laptop", "quantity": 1, "price": 999.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 999.99, "method": "card"}},
        ],
        difficulty=2, task_type="create_and_pay"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "Laptop", "quantity": 1, "price": 999.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 500.00, "method": "card"}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_5_wrong_payment_amount", result, expect_success=False)


def test_wrong_6_skip_step():
    """Trying to ship without paying first — state machine violation."""
    task = Task(
        instruction="Create, pay, ship",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "Mouse", "quantity": 1, "price": 29.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 29.99, "method": "card"}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": "123 Street", "carrier": "dhl"}},
        ],
        difficulty=2, task_type="create_pay_ship"
    )
    # Agent skips pay, goes directly to ship
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "Mouse", "quantity": 1, "price": 29.99}

ACTION: POST /orders/PLACEHOLDER/ship
BODY: {"order_id": "PLACEHOLDER", "address": "123 Street", "carrier": "dhl"}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_6_skip_step", result, expect_success=False)


def test_wrong_7_invalid_carrier():
    """Invalid carrier should be rejected by API."""
    task = Task(
        instruction="Create, pay, ship",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "Keyboard", "quantity": 1, "price": 49.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 49.99, "method": "card"}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": "456 Ave", "carrier": "dhl"}},
        ],
        difficulty=2, task_type="create_pay_ship"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "Keyboard", "quantity": 1, "price": 49.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 49.99, "method": "card"}

ACTION: POST /orders/PLACEHOLDER/ship
BODY: {"order_id": "PLACEHOLDER", "address": "456 Ave", "carrier": "amazon_logistics"}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_7_invalid_carrier", result, expect_success=False)


def test_wrong_8_extra_steps():
    """More actions than expected — step count mismatch."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "Phone Case", "quantity": 1, "price": 19.99}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "Phone Case", "quantity": 1, "price": 19.99}

ACTION: GET /orders/PLACEHOLDER

ACTION: POST /orders/PLACEHOLDER/cancel"""
    result = execute_and_verify(task, agent_output)
    check("wrong_8_extra_steps", result, expect_success=False)


def test_wrong_9_invalid_json_body():
    """Malformed JSON body should fail."""
    task = Task(
        instruction="Create order",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "RAM 16GB", "quantity": 1, "price": 54.99}}
        ],
        difficulty=1, task_type="create_order"
    )
    agent_output = """ACTION: POST /orders
BODY: {customer_name: Test, product: RAM, quantity: 1}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_9_invalid_json_body", result, expect_success=False)


def test_wrong_10_invalid_payment_method():
    """Invalid payment method (bitcoin instead of card/bank_transfer)."""
    task = Task(
        instruction="Create and pay",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": "Test", "product": "Charger", "quantity": 1, "price": 24.99}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": 24.99, "method": "card"}},
        ],
        difficulty=2, task_type="create_and_pay"
    )
    agent_output = """ACTION: POST /orders
BODY: {"customer_name": "Test", "product": "Charger", "quantity": 1, "price": 24.99}

ACTION: POST /orders/PLACEHOLDER/pay
BODY: {"order_id": "PLACEHOLDER", "amount": 24.99, "method": "bitcoin"}"""
    result = execute_and_verify(task, agent_output)
    check("wrong_10_invalid_payment_method", result, expect_success=False)


# ============================================================
# RUNNER
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  VERIFIER UNIT TESTS — RLVR Reward Signal Validation")
    print("=" * 60)

    print("\n--- 10 CORRECT outputs (expect reward=1.0) ---")
    test_correct_1_create_order()
    test_correct_2_create_and_check()
    test_correct_3_create_and_cancel()
    test_correct_4_create_and_pay()
    test_correct_5_create_pay_ship()
    test_correct_6_pay_and_refund()
    test_correct_7_full_delivery()
    test_correct_8_full_lifecycle_refund()
    test_correct_9_parse_format()
    test_correct_10_single_item_quantity()

    print("\n--- 10 INCORRECT outputs (expect reward=0.0) ---")
    test_wrong_1_empty_output()
    test_wrong_2_no_action_keyword()
    test_wrong_3_wrong_method()
    test_wrong_4_wrong_path()
    test_wrong_5_wrong_payment_amount()
    test_wrong_6_skip_step()
    test_wrong_7_invalid_carrier()
    test_wrong_8_extra_steps()
    test_wrong_9_invalid_json_body()
    test_wrong_10_invalid_payment_method()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed out of {PASSED+FAILED}")
    if FAILED == 0:
        print("  ✓ Verifier reward signal is RELIABLE")
    else:
        print(f"  ✗ {FAILED} tests failed — verifier needs fixing")
    print(f"{'='*60}")

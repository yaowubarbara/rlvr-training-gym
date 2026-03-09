"""
API Training Gym - Module 2: Procedural Task Generator
Generates infinite task variations with known correct action sequences.
"""
import random
from dataclasses import dataclass

# ===== Product catalog for variation =====
PRODUCTS = [
    ("Laptop", 999.99), ("Keyboard", 49.99), ("Mouse", 29.99),
    ("Monitor", 399.99), ("Headphones", 79.99), ("Webcam", 59.99),
    ("USB Cable", 9.99), ("SSD 1TB", 89.99), ("RAM 16GB", 54.99),
    ("Phone Case", 19.99), ("Tablet", 449.99), ("Charger", 24.99),
]

CUSTOMERS = [
    "Alice Martin", "Bob Chen", "Claire Dubois", "David Kim",
    "Emma Wilson", "François Petit", "Grace Liu", "Henri Bernard",
    "Isabelle Wang", "Jean Moreau", "Kenji Tanaka", "Laura Schmidt",
]

CARRIERS = ["dhl", "fedex", "ups"]
PAYMENT_METHODS = ["card", "bank_transfer"]
ADDRESSES = [
    "123 Rue de Rivoli, Paris 75001",
    "456 Keizersgracht, Amsterdam 1016",
    "789 Oxford Street, London W1D 1BS",
    "321 Friedrichstraße, Berlin 10117",
    "654 Via Roma, Milan 20121",
]

REFUND_REASONS = [
    "Product defective",
    "Wrong item received",
    "Changed my mind",
    "Better price elsewhere",
    "Arrived too late",
]


@dataclass
class Task:
    """A task for the LLM agent to complete."""
    instruction: str           # Natural language task description
    expected_actions: list     # Correct sequence of API calls
    difficulty: int            # 1=easy, 2=medium, 3=hard
    task_type: str             # Category name


def _random_order_params():
    product, price = random.choice(PRODUCTS)
    quantity = random.randint(1, 5)
    customer = random.choice(CUSTOMERS)
    return customer, product, quantity, price


def generate_task(difficulty: int = None) -> Task:
    """Generate a random task at the specified difficulty level."""
    if difficulty is None:
        difficulty = random.choice([1, 1, 1, 2, 2, 3])  # weighted toward easier

    generators = {
        1: [_task_create_order, _task_check_order, _task_cancel_new_order],
        2: [_task_create_and_pay, _task_full_flow_to_ship, _task_pay_and_refund],
        3: [_task_full_delivery, _task_complex_multi_order, _task_create_pay_cancel_refund],
    }

    gen_fn = random.choice(generators[difficulty])
    return gen_fn()


# ===== Difficulty 1: Single-step tasks =====

def _task_create_order() -> Task:
    customer, product, qty, price = _random_order_params()
    return Task(
        instruction=f"Create an order for {customer}: {qty}x {product} at ${price} each.",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}}
        ],
        difficulty=1,
        task_type="create_order",
    )

def _task_check_order() -> Task:
    customer, product, qty, price = _random_order_params()
    return Task(
        instruction=f"Create an order for {customer}: {qty}x {product} at ${price}. Then check its status.",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "GET", "path": "/orders/{order_id}"},
        ],
        difficulty=1,
        task_type="create_and_check",
    )

def _task_cancel_new_order() -> Task:
    customer, product, qty, price = _random_order_params()
    return Task(
        instruction=f"Create an order for {customer}: {qty}x {product} at ${price}. Then cancel it immediately.",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/cancel"},
        ],
        difficulty=1,
        task_type="create_and_cancel",
    )


# ===== Difficulty 2: Multi-step tasks =====

def _task_create_and_pay() -> Task:
    customer, product, qty, price = _random_order_params()
    total = round(qty * price, 2)
    method = random.choice(PAYMENT_METHODS)
    return Task(
        instruction=f"Create an order for {customer}: {qty}x {product} at ${price}. Pay with {method}.",
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": total, "method": method}},
        ],
        difficulty=2,
        task_type="create_and_pay",
    )

def _task_full_flow_to_ship() -> Task:
    customer, product, qty, price = _random_order_params()
    total = round(qty * price, 2)
    method = random.choice(PAYMENT_METHODS)
    carrier = random.choice(CARRIERS)
    address = random.choice(ADDRESSES)
    return Task(
        instruction=(f"Process order for {customer}: {qty}x {product} at ${price}. "
                     f"Pay with {method}, then ship via {carrier} to {address}."),
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": total, "method": method}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": address, "carrier": carrier}},
        ],
        difficulty=2,
        task_type="create_pay_ship",
    )

def _task_pay_and_refund() -> Task:
    customer, product, qty, price = _random_order_params()
    total = round(qty * price, 2)
    method = random.choice(PAYMENT_METHODS)
    reason = random.choice(REFUND_REASONS)
    return Task(
        instruction=(f"Create order for {customer}: {qty}x {product} at ${price}. "
                     f"Pay with {method}, then refund because: {reason}"),
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": total, "method": method}},
            {"method": "POST", "path": "/orders/{order_id}/refund",
             "body": {"order_id": "{order_id}", "reason": reason}},
        ],
        difficulty=2,
        task_type="pay_and_refund",
    )


# ===== Difficulty 3: Complex multi-step tasks =====

def _task_full_delivery() -> Task:
    customer, product, qty, price = _random_order_params()
    total = round(qty * price, 2)
    method = random.choice(PAYMENT_METHODS)
    carrier = random.choice(CARRIERS)
    address = random.choice(ADDRESSES)
    return Task(
        instruction=(f"Complete full order lifecycle for {customer}: {qty}x {product} at ${price}. "
                     f"Pay with {method}, ship via {carrier} to {address}, then confirm delivery."),
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": total, "method": method}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": address, "carrier": carrier}},
            {"method": "POST", "path": "/orders/{order_id}/deliver"},
        ],
        difficulty=3,
        task_type="full_delivery",
    )

def _task_complex_multi_order() -> Task:
    c1, p1, q1, pr1 = _random_order_params()
    c2, p2, q2, pr2 = _random_order_params()
    t1 = round(q1 * pr1, 2)
    t2 = round(q2 * pr2, 2)
    m1 = random.choice(PAYMENT_METHODS)
    return Task(
        instruction=(f"Create TWO orders:\n"
                     f"  Order A: {c1}, {q1}x {p1} at ${pr1} — pay with {m1}\n"
                     f"  Order B: {c2}, {q2}x {p2} at ${pr2} — cancel immediately\n"),
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": c1, "product": p1, "quantity": q1, "price": pr1}},
            {"method": "POST", "path": "/orders/{order_id_1}/pay",
             "body": {"order_id": "{order_id_1}", "amount": t1, "method": m1}},
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": c2, "product": p2, "quantity": q2, "price": pr2}},
            {"method": "POST", "path": "/orders/{order_id_2}/cancel"},
        ],
        difficulty=3,
        task_type="multi_order",
    )

def _task_create_pay_cancel_refund() -> Task:
    customer, product, qty, price = _random_order_params()
    total = round(qty * price, 2)
    method = random.choice(PAYMENT_METHODS)
    carrier = random.choice(CARRIERS)
    address = random.choice(ADDRESSES)
    reason = random.choice(REFUND_REASONS)
    return Task(
        instruction=(f"Full lifecycle with refund: {customer} orders {qty}x {product} at ${price}. "
                     f"Pay with {method}, ship via {carrier} to {address}, confirm delivery, "
                     f"then process refund: '{reason}'"),
        expected_actions=[
            {"method": "POST", "path": "/orders",
             "body": {"customer_name": customer, "product": product,
                      "quantity": qty, "price": price}},
            {"method": "POST", "path": "/orders/{order_id}/pay",
             "body": {"order_id": "{order_id}", "amount": total, "method": method}},
            {"method": "POST", "path": "/orders/{order_id}/ship",
             "body": {"order_id": "{order_id}", "address": address, "carrier": carrier}},
            {"method": "POST", "path": "/orders/{order_id}/deliver"},
            {"method": "POST", "path": "/orders/{order_id}/refund",
             "body": {"order_id": "{order_id}", "reason": reason}},
        ],
        difficulty=3,
        task_type="full_lifecycle_with_refund",
    )


# ===== Curriculum Strategies =====

def generate_curriculum(n_tasks: int, start_difficulty: int = 1) -> list[Task]:
    """Generate a fixed-ratio curriculum (legacy).
    First 40% at start_difficulty, next 35% at +1, last 25% at +2.
    """
    tasks = []
    for i in range(n_tasks):
        progress = i / n_tasks
        if progress < 0.4:
            diff = start_difficulty
        elif progress < 0.75:
            diff = min(start_difficulty + 1, 3)
        else:
            diff = min(start_difficulty + 2, 3)
        tasks.append(generate_task(diff))
    return tasks


import math

class BoltzmannCurriculum:
    """Adaptive curriculum using Boltzmann exploration.

    Instead of fixed ratios, dynamically adjust difficulty distribution
    based on the model's current reward per difficulty level.

    P(difficulty=d) ∝ exp(-reward_d / τ)

    Low reward at difficulty d → higher probability of sampling d
    (focus on what the model struggles with)

    τ (temperature) controls exploration:
    - High τ → uniform sampling (explore all difficulties)
    - Low τ → focus on hardest (exploit weaknesses)
    - τ decays over training (anneal from exploration to exploitation)
    """

    def __init__(self, tau_start: float = 2.0, tau_end: float = 0.3, difficulties: list[int] = None):
        self.difficulties = difficulties or [1, 2, 3]
        self.tau_start = tau_start
        self.tau_end = tau_end
        # Running average rewards per difficulty
        self.reward_history = {d: [] for d in self.difficulties}
        self.avg_rewards = {d: 0.5 for d in self.difficulties}  # Prior: assume 50% success

    def get_tau(self, progress: float) -> float:
        """Anneal temperature from tau_start to tau_end."""
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def get_difficulty_probs(self, progress: float) -> dict[int, float]:
        """Compute Boltzmann probability for each difficulty level."""
        tau = self.get_tau(progress)

        # Invert rewards: lower reward = higher sampling probability
        # Use (1 - avg_reward) so struggling difficulties get more weight
        logits = {}
        for d in self.difficulties:
            logits[d] = (1.0 - self.avg_rewards[d]) / tau

        # Softmax normalization
        max_logit = max(logits.values())
        exp_logits = {d: math.exp(l - max_logit) for d, l in logits.items()}
        total = sum(exp_logits.values())
        probs = {d: exp_logits[d] / total for d in self.difficulties}
        return probs

    def sample_difficulty(self, progress: float) -> int:
        """Sample a difficulty level using Boltzmann distribution."""
        probs = self.get_difficulty_probs(progress)
        r = random.random()
        cumulative = 0.0
        for d in self.difficulties:
            cumulative += probs[d]
            if r <= cumulative:
                return d
        return self.difficulties[-1]

    def update_reward(self, difficulty: int, reward: float):
        """Update running average reward for a difficulty level."""
        self.reward_history[difficulty].append(reward)
        # Exponential moving average (recent results matter more)
        alpha = 0.1
        self.avg_rewards[difficulty] = (
            alpha * reward + (1 - alpha) * self.avg_rewards[difficulty]
        )

    def generate_batch(self, batch_size: int, progress: float) -> list[Task]:
        """Generate a batch of tasks using Boltzmann sampling."""
        tasks = []
        probs = self.get_difficulty_probs(progress)
        for _ in range(batch_size):
            diff = self.sample_difficulty(progress)
            tasks.append(generate_task(diff))
        return tasks

    def stats(self) -> str:
        """Return curriculum stats for logging."""
        counts = {d: len(h) for d, h in self.reward_history.items()}
        return (f"Curriculum stats — "
                f"Rewards: {self.avg_rewards} | "
                f"Counts: {counts}")


if __name__ == "__main__":
    # Demo: generate 5 tasks at each difficulty
    for diff in [1, 2, 3]:
        print(f"\n=== Difficulty {diff} ===")
        for _ in range(3):
            task = generate_task(diff)
            print(f"  [{task.task_type}] {task.instruction[:80]}...")
            print(f"    Steps: {len(task.expected_actions)}")

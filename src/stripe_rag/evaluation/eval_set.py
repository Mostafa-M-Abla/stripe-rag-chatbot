"""25 hand-crafted Q&A pairs covering all 4 Stripe doc sections."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalPair:
    question: str
    expected_url_prefix: str
    keywords: list[str]
    section: str  # payments | billing | connect | api | webhooks


EVAL_SET: list[EvalPair] = [
    # ── Payments (6) ──────────────────────────────────────────────────────────
    EvalPair(
        question="How do I create a PaymentIntent?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["PaymentIntent", "client_secret", "confirm"],
        section="payments",
    ),
    EvalPair(
        question="What payment methods does Stripe support?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["card", "bank", "method"],
        section="payments",
    ),
    EvalPair(
        question="How do I handle 3D Secure authentication with Stripe?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["3D Secure", "authentication", "next_action"],
        section="payments",
    ),
    EvalPair(
        question="How do I accept payments with Stripe Elements?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["Elements", "card", "mount"],
        section="payments",
    ),
    EvalPair(
        question="How do I issue a refund for a payment?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["refund", "Refund", "charge"],
        section="payments",
    ),
    EvalPair(
        question="What is the difference between a charge and a PaymentIntent?",
        expected_url_prefix="https://docs.stripe.com/payments",
        keywords=["PaymentIntent", "Charge"],
        section="payments",
    ),
    # ── Billing (6) ───────────────────────────────────────────────────────────
    EvalPair(
        question="How do I create a subscription with Stripe Billing?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["subscription", "customer", "price"],
        section="billing",
    ),
    EvalPair(
        question="What is a Stripe Price object and how does it differ from a Plan?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["Price", "amount", "recurring"],
        section="billing",
    ),
    EvalPair(
        question="How do I cancel a subscription in Stripe?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["cancel", "subscription", "cancel_at"],
        section="billing",
    ),
    EvalPair(
        question="How does metered billing work in Stripe?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["metered", "usage", "meter"],
        section="billing",
    ),
    EvalPair(
        question="How do I apply a coupon or discount to a subscription?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["coupon", "discount", "promotion"],
        section="billing",
    ),
    EvalPair(
        question="How do free trials work with Stripe subscriptions?",
        expected_url_prefix="https://docs.stripe.com/billing",
        keywords=["trial", "trial_end", "trial_period"],
        section="billing",
    ),
    # ── Connect (6) ───────────────────────────────────────────────────────────
    EvalPair(
        question="What is Stripe Connect and when should I use it?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["Connect", "platform", "marketplace"],
        section="connect",
    ),
    EvalPair(
        question="What are the different Stripe Connect account types?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["Standard", "Express", "Custom"],
        section="connect",
    ),
    EvalPair(
        question="How do I create a connected account in Stripe?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["Account", "create", "connected"],
        section="connect",
    ),
    EvalPair(
        question="How do payouts work for connected accounts?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["payout", "transfer", "bank"],
        section="connect",
    ),
    EvalPair(
        question="What is the difference between destination charges and direct charges in Connect?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["destination", "direct", "charge"],
        section="connect",
    ),
    EvalPair(
        question="How do I handle onboarding for connected accounts with Stripe Connect?",
        expected_url_prefix="https://docs.stripe.com/connect",
        keywords=["onboarding", "AccountLink", "verification"],
        section="connect",
    ),
    # ── API & Webhooks (7) ────────────────────────────────────────────────────
    EvalPair(
        question="How do I authenticate requests to the Stripe API?",
        expected_url_prefix="https://docs.stripe.com/api",
        keywords=["API key", "secret", "Authorization"],
        section="api",
    ),
    EvalPair(
        question="What are Stripe API rate limits?",
        expected_url_prefix="https://docs.stripe.com/api",
        keywords=["rate limit", "429", "request"],
        section="api",
    ),
    EvalPair(
        question="How do I verify a Stripe webhook signature?",
        expected_url_prefix="https://docs.stripe.com/webhooks",
        keywords=["signature", "Stripe-Signature", "construct_event"],
        section="webhooks",
    ),
    EvalPair(
        question="What are Stripe webhooks and how do event types work?",
        expected_url_prefix="https://docs.stripe.com/webhooks",
        keywords=["webhook", "event", "endpoint"],
        section="webhooks",
    ),
    EvalPair(
        question="How do I handle errors from the Stripe API?",
        expected_url_prefix="https://docs.stripe.com/api",
        keywords=["error", "code", "StripeError"],
        section="api",
    ),
    EvalPair(
        question="What are idempotency keys and why should I use them?",
        expected_url_prefix="https://docs.stripe.com/api",
        keywords=["idempotency", "Idempotency-Key", "retry"],
        section="api",
    ),
    EvalPair(
        question="How do I paginate through results in the Stripe API?",
        expected_url_prefix="https://docs.stripe.com/api",
        keywords=["pagination", "limit", "starting_after"],
        section="api",
    ),
]

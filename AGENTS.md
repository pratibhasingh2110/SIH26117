# AGENTS.md — Agent Runtime

## 1. Project Objective

Build a lightweight, provider-independent **Agent Runtime from scratch in Python**.

The runtime will eventually be used underneath a **Level-0 Agent Router**.

The target architecture is:

User Request
    ↓
Level-0 Router
    ↓
Agent Selection
    ↓
Agent
    ↓
Agent Runtime
    ↓
LLM + Tools
    ↓
Execution Result

The runtime is the execution engine.

The router is the selection engine.

These responsibilities MUST remain separate.

---

# 2. Core Architecture

The system consists of these major layers:

```text
┌─────────────────────────────┐
│       Level-0 Router        │
│                             │
│  Selects the appropriate    │
│  Agent for a task           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│           Agent             │
│                             │
│ name                        │
│ instructions                │
│ LLM provider                │
│ tools                       │
│ configuration               │
└──────────────┬──────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│              Agent Runtime               │
│                                          │
│ State                                    │
│ Context                                  │
│ LLM Provider                             │
│ Response Parser                          │
│ Action Dispatcher                        │
│ Policy / Guardrails                      │
│ Tool Executor                            │
│ Events / Tracing                         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
                Tools
"""Where an agent runtime is allowed to have a name.

One module per product, each exporting `NAME` -- what `--runtime` is given --
`PRODUCT` -- what it wraps, the words the contract must never use -- and
`Runtime`, a class satisfying `qmcp.integrations.agents.AgentRuntime`.
`qmcp.integrations.agents` discovers them from this package by those two
attributes and imports nothing here by name, so adding a runtime is adding a
file, and the contract, the service and the docs stay free of the product.
"""

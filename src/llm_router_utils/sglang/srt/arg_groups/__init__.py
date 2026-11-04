"""Stub for sglang.srt.arg_groups.

Upstream's ``arg_groups`` package declares every ``ServerArgs`` field once and
derives the CLI flags, per-role config namespaces and resolution bookkeeping
from it. That belongs to the server-launch path, which this repo does not
carry. See the sibling modules for the two shims the retained parser layer
actually calls.
"""

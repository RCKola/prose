"""Thin HF-model wrappers. Each wrapper owns its model/processor and exposes
a small, task-focused method. Call .close() (or use as a context manager) to
release GPU memory between stages."""

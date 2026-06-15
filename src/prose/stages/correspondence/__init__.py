"""Cross-scan instance correspondence (paper Sec. 3.2).

Given the two per-subscan scene graphs, this stage produces cross-scan
instance correspondences. It measures each instance's height along the gravity
axis, splits instances into K quantile height bins (with overlap), and queries
the VLM separately within each bin over shared-namespace Set-of-Marks crop
panels — so a pair of marker numbers is directly a candidate match. A cross-bin
pass recovers matches that never shared a bin, and a two-pass *same?* /
*different?* double-check verifies each candidate.

The pipeline is composed from small phase objects (filter, prompt, parser,
resolver, postprocess) driven by ``pipeline.Stage4Pipeline``; the per-bin loop
lives in ``visuals.blocking_meta.run_blocking_pipeline``.
"""

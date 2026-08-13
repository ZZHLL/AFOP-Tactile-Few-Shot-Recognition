# Evaluation Protocol

This module creates deterministic train/validation/test splits and shared
episodic manifests. Every model receives the same stored support/query indices;
the validator checks index bounds, labels, and support/query disjointness.

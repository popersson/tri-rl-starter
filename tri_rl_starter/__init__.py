"""Lean triangular-mesh improvement starter package."""

from .geometry import CASE_NAMES, generate_case_mesh
from .mesh import TriMesh

__all__ = ["CASE_NAMES", "TriMesh", "generate_case_mesh"]

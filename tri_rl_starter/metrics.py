"""Metrics for triangular mesh improvement."""

import math

import numpy as np


QUALITY_FLOOR = 1.0e-14


def triangle_areas(points, triangles, signed=True):
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    xy = points[triangles]
    area = 0.5 * (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])
    )
    return area if signed else np.abs(area)


def orient_triangles_ccw(points, triangles):
    triangles = np.asarray(triangles, dtype=int).copy()
    area = triangle_areas(points, triangles, signed=True)
    flip = area < 0.0
    if np.any(flip):
        triangles[flip, 1], triangles[flip, 2] = (
            triangles[flip, 2].copy(),
            triangles[flip, 1].copy(),
        )
    return triangles


def triangle_mean_ratio(points, triangles):
    """Return mean-ratio quality in [0, 1] for each triangle."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    xy = points[triangles]
    e01 = xy[:, 1] - xy[:, 0]
    e12 = xy[:, 2] - xy[:, 1]
    e20 = xy[:, 0] - xy[:, 2]
    lengths2 = (
        np.einsum("ij,ij->i", e01, e01)
        + np.einsum("ij,ij->i", e12, e12)
        + np.einsum("ij,ij->i", e20, e20)
    )
    area = triangle_areas(points, triangles, signed=True)
    quality = np.zeros(len(triangles), dtype=float)
    ok = (area > QUALITY_FLOOR) & (lengths2 > QUALITY_FLOOR)
    quality[ok] = 4.0 * math.sqrt(3.0) * area[ok] / lengths2[ok]
    return np.clip(quality, 0.0, 1.0)


def quality_summary(points, triangles):
    q = triangle_mean_ratio(points, triangles)
    if len(q) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "p10": 0.0,
            "inverted": 0,
        }
    return {
        "count": int(len(q)),
        "min": float(q.min()),
        "mean": float(q.mean()),
        "p10": float(np.percentile(q, 10.0)),
        "inverted": int(np.count_nonzero(q <= QUALITY_FLOOR)),
    }


def degree_deviation(degrees, ideal_degrees):
    degrees = np.asarray(degrees, dtype=float)
    ideal_degrees = np.asarray(ideal_degrees, dtype=float)
    signed = degrees - ideal_degrees
    absolute = np.abs(signed)
    return {
        "signed": signed,
        "absolute": absolute,
        "l1": float(absolute.sum()),
        "l2": float(np.linalg.norm(signed)),
        "l2sq": float(np.dot(signed, signed)),
        "linf": float(absolute.max()) if len(absolute) else 0.0,
    }

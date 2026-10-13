"""Versioned, unit-normalized vectors; independent of the unchanged M1 contract."""

import math
from typing import List, Literal, Optional

import numpy as np
from pydantic import Field, model_validator

from ..m3.models import NormalizedProduct
from ..schemas import Contract


class Vector(Contract):
    model: str = Field(min_length=1)
    modality: Literal["text", "image"]
    values: List[float] = Field(min_length=1)

    @model_validator(mode="after")
    def unit_vector(self):
        if not all(math.isfinite(v) for v in self.values):
            raise ValueError("vector contains nonfinite values")
        norm = float(np.linalg.norm(self.values))
        if not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("vector must have unit L2 norm, including nonzero magnitude")
        return self


def vector(model, modality, values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12:
        raise ValueError("embedding must be a finite nonzero 1D vector")
    return Vector(model=model, modality=modality, values=(a / np.linalg.norm(a)).tolist())


def cosine(a: Vector, b: Vector):
    if a.model != b.model or len(a.values) != len(b.values):
        raise ValueError("incompatible embedding model/revision/preprocessing or dimension")
    return float(np.clip(np.dot(a.values, b.values), -1.0, 1.0))


class IndexedProduct(Contract):
    product: NormalizedProduct
    vector: Vector
    encoding_reason: str


class IndexIssue(Contract):
    product_id: Optional[str] = None
    reason: str


class ProductIndex(Contract):
    schema_version: Literal["m5.1"] = "m5.1"
    model: str
    dimension: int = Field(ge=1)
    items: List[IndexedProduct] = Field(default_factory=list)
    issues: List[IndexIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def compatible(self):
        seen = set()
        for item in self.items:
            key = (item.product.source_type, item.product.product_id)
            if key in seen:
                raise ValueError("duplicate indexed identity")
            seen.add(key)
            if item.vector.model != self.model or len(item.vector.values) != self.dimension:
                raise ValueError("index has incompatible vectors")
        return self

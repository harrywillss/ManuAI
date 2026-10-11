"""Class set and label mapping.

The class list lives in `types.yaml` (English label → scientific names). Everything
identified but not listed becomes `Other`, so the model can say "not one I know".
Unidentified archive labels are dropped entirely, because they could be any species
(including target classes).
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yaml

from .config import ARCHIVE_SOURCES, TYPES_PATH, archive_source_dir

OTHER = "Other"

# Archive labels that do not identify a species. `parake` is "Psittaciformes sp."
# (an unidentified parrot — could be kākā, kea or kākāriki). `badbird` is a junk label.
# `spybird` only appears as a secondary label and means "unidentified bird".
EXCLUDED_CODES: frozenset[str] = frozenset({"parake", "badbird", "spybird"})


def _species_key(name: str) -> str:
    """Lower-cased binomial ('Genus species'), ignoring subspecies and stray spaces."""
    parts = str(name).replace("_", " ").lower().split()
    return " ".join(parts[:2])


@lru_cache(maxsize=1)
def load_types() -> dict[str, list[str]]:
    with open(TYPES_PATH) as f:
        types = yaml.safe_load(f)
    if not isinstance(types, dict) or not types:
        raise ValueError(f"{TYPES_PATH} is empty or malformed")
    if OTHER in types:
        raise ValueError(f"'{OTHER}' is reserved; do not list it in types.yaml")
    return types


def class_names(include_other: bool = True) -> list[str]:
    """Stable, sorted class order. `Other` is always last."""
    names = sorted(load_types().keys())
    return names + [OTHER] if include_other else names


@lru_cache(maxsize=1)
def naming_map() -> pd.DataFrame:
    """eBird code → CommonName / ScientificName / ExtraName, merged over all sources."""
    frames = [pd.read_csv(archive_source_dir(s) / "bird_naming_map.csv") for s in ARCHIVE_SOURCES]
    nm = pd.concat(frames).drop_duplicates("eBird").set_index("eBird")
    nm["ScientificName"] = nm["ScientificName"].astype(str).str.strip()
    return nm


@lru_cache(maxsize=None)
def ebird_to_class(code: str) -> str | None:
    """Map an archive eBird code to a class name, `Other`, or None (excluded)."""
    if code in EXCLUDED_CODES:
        return None
    lookup = {_species_key(s): label for label, names in load_types().items() for s in names}
    nm = naming_map()
    if code in nm.index:
        sci = nm.loc[code, "ScientificName"]
        if _species_key(sci) in lookup:
            return lookup[_species_key(sci)]
    # Identified species not in the class list (including codes missing from the
    # naming map, which are real species such as Chatham oystercatcher), and Tree wētā.
    return OTHER

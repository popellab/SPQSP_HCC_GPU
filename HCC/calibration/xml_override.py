"""Apply dot-path overrides to the HCC parameter XML and write to a new file.

Example:
    apply_overrides("param_all_test.xml", {"ABM.TCell.moveSteps": 42}, "tmp.xml")

Only leaf-text values are modified. Missing paths raise KeyError.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping


def apply_overrides(
    base_xml: str | Path,
    overrides: Mapping[str, float | int | str],
    out_path: str | Path,
) -> Path:
    tree = ET.parse(str(base_xml))
    root = tree.getroot()  # <Param>

    for dotted, value in overrides.items():
        parts = dotted.split(".")
        # The outer <Param> is implicit; overrides are relative to it.
        node = root
        for p in parts:
            child = node.find(p)
            if child is None:
                raise KeyError(
                    f"XML path not found: {dotted} (missing element '{p}' under <{node.tag}>)"
                )
            node = child
        node.text = _fmt(value)

    out_path = Path(out_path)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return str(v)

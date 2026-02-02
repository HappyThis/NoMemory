from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillMetadata:
    skill_id: str
    name: str
    description: str


@dataclass(frozen=True)
class Skill:
    name: str
    root: Path
    metadata: SkillMetadata
    text: str


class SkillNotFoundError(FileNotFoundError):
    pass


def _parse_frontmatter(md_text: str) -> tuple[dict[str, str], str]:
    """
    Parse a minimal YAML-frontmatter header:

    ---
    name: xxx
    description: yyy
    ---

    Returns (meta, body_without_frontmatter).
    """
    if not md_text.startswith("---"):
        return ({}, md_text)
    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ({}, md_text)

    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return ({}, md_text)

    meta: dict[str, str] = {}
    for raw in lines[1:end_idx]:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")

    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return (meta, body)


def list_skill_metadata(*, skills_dir: str = "skills") -> list[SkillMetadata]:
    """
    Progressive loading level 1: enumerate skills and read only their metadata.
    """
    root = Path(skills_dir)
    if not root.exists():
        return []

    out: list[SkillMetadata] = []
    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        md_text = skill_md.read_text(encoding="utf-8")
        meta, _ = _parse_frontmatter(md_text)
        out.append(
            SkillMetadata(
                skill_id=child.name,
                name=str(meta.get("name") or child.name),
                description=str(meta.get("description") or ""),
            )
        )
    return out


def load_skill(skill_name: str, *, skills_dir: str = "skills") -> Skill:
    """
    Progressive loading level 2: load skill instructions (SKILL.md) for a chosen skill.
    Note: does NOT auto-load any resources under scripts/; those should be loaded on demand.
    """
    root = Path(skills_dir) / skill_name
    skill_md = root / "SKILL.md"
    if not skill_md.exists():
        raise SkillNotFoundError(f"Skill not found: {skill_name}")

    md_text = skill_md.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(md_text)
    metadata = SkillMetadata(
        skill_id=skill_name,
        name=str(meta.get("name") or skill_name),
        description=str(meta.get("description") or ""),
    )
    return Skill(name=skill_name, root=root, metadata=metadata, text=body)

#!/usr/bin/env python3
"""Resolve FantasyWorld text prompts for curated_set images.

Primary source: data/curated_set/metadata.json (nested photorealistic/stylized
blocks already carry WorldScore-style prompts). Falls back to a lightweight
template built from scene metadata when a file is not listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STYLE_SUFFIX = {
    "photorealistic": "",
    "anime": ", rendered in a detailed anime style",
    "ukiyo-e": ", depicted in a traditional Japanese ukiyo-e woodblock style",
    "chinese_ink_painting": ", painted in classical Chinese ink wash style",
    "impressionism": ", rendered in an Impressionist painting style",
    "post_Impressionism": ", rendered in a Post-Impressionist painting style",
    "cyberpunk_game": ", in a neon cyberpunk game-art style",
    "minecraft_game": ", in a blocky Minecraft game style",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_metadata_path() -> Path:
    return _repo_root() / "data" / "curated_set" / "metadata.json"


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_index(metadata: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (folder, filename) -> {prompt, style, content, category, scene_type}."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in metadata.get("samples", []):
        for folder in ("photorealistic", "stylized"):
            block = sample.get(folder)
            if not isinstance(block, dict):
                continue
            filename = block.get("filename")
            if not filename:
                continue
            prompt = (block.get("prompt") or "").strip()
            style = block.get("style") or sample.get("style") or folder
            index[(folder, filename)] = {
                "prompt": prompt,
                "style": style,
                "content": sample.get("content") or "",
                "category": sample.get("category") or "",
                "scene_type": sample.get("scene_type") or "",
            }
    return index


def template_prompt(
    content: str = "",
    category: str = "",
    scene_type: str = "",
    style: str = "photorealistic",
) -> str:
    """Fallback prompt when metadata has no explicit prompt string."""
    subject = content.strip() or category.strip() or "scene"
    # Prefer "In the X, ..." WorldScore / FantasyWorld phrasing.
    if "," in subject:
        title, rest = subject.split(",", 1)
        title = title.strip()
        details = rest.strip().rstrip(".")
        base = f"In the {title}, {details}."
    else:
        place = scene_type.strip() or "scene"
        base = f"In this {place} {subject}, the environment is richly detailed with natural lighting and clear spatial layout."

    suffix = STYLE_SUFFIX.get(style, "")
    if suffix and not style.startswith("photo"):
        # Avoid double-appending if the template already mentions the style.
        if style.replace("_", " ").lower() not in base.lower():
            base = base.rstrip(".") + suffix + "."
    return " ".join(base.split())


def decorate_prompt(prompt: str, style: str, append_style: bool) -> str:
    prompt = " ".join(prompt.split()).strip()
    if not append_style:
        return prompt
    suffix = STYLE_SUFFIX.get(style, "")
    if not suffix:
        return prompt
    style_token = style.replace("_", " ").lower()
    if style_token in prompt.lower():
        return prompt
    return prompt.rstrip(".") + suffix + "."


def resolve_prompt_for_image(
    image_path: Path,
    index: dict[tuple[str, str], dict[str, Any]],
    append_style: bool = True,
) -> tuple[str, dict[str, Any]]:
    image_path = image_path.resolve()
    folder = image_path.parent.name
    filename = image_path.name
    meta = index.get((folder, filename))

    if meta and meta.get("prompt"):
        prompt = decorate_prompt(meta["prompt"], meta.get("style", "photorealistic"), append_style)
        return prompt, meta

    if meta:
        prompt = template_prompt(
            content=meta.get("content", ""),
            category=meta.get("category", ""),
            scene_type=meta.get("scene_type", ""),
            style=meta.get("style", "photorealistic"),
        )
        prompt = decorate_prompt(prompt, meta.get("style", "photorealistic"), append_style)
        return prompt, meta

    # Last resort: invent from path hints.
    style_guess = folder if folder in STYLE_SUFFIX else "photorealistic"
    prompt = template_prompt(
        content=image_path.stem.replace("_", " "),
        style=style_guess,
    )
    prompt = decorate_prompt(prompt, style_guess, append_style)
    return prompt, {"style": style_guess, "source": "path_fallback"}


def iter_images(root: Path) -> list[Path]:
    images = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate / look up FantasyWorld prompts for curated_set images."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Single image path; prints the prompt to stdout.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Recursive image root for batch mode (writes sibling .prompt.txt files).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(_default_metadata_path()),
        help="Path to curated_set metadata.json",
    )
    parser.add_argument(
        "--append_style",
        action="store_true",
        default=True,
        help="Append style hint for non-photorealistic images (default: on).",
    )
    parser.add_argument(
        "--no_append_style",
        action="store_true",
        help="Disable style suffix decoration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .prompt.txt files in batch mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON with prompt + metadata instead of plain text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    append_style = args.append_style and not args.no_append_style

    metadata_path = Path(args.metadata)
    if not metadata_path.is_file():
        print(f"[generate_prompt] metadata not found: {metadata_path}", file=sys.stderr)
        return 1

    index = build_prompt_index(load_metadata(metadata_path))

    if args.image_path:
        image_path = Path(args.image_path)
        if not image_path.is_file():
            print(f"[generate_prompt] image not found: {image_path}", file=sys.stderr)
            return 1
        prompt, meta = resolve_prompt_for_image(image_path, index, append_style=append_style)
        if args.json:
            print(json.dumps({"image_path": str(image_path), "prompt": prompt, "meta": meta}, ensure_ascii=False))
        else:
            print(prompt)
        return 0

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"[generate_prompt] data_dir not found: {data_dir}", file=sys.stderr)
            return 1
        images = iter_images(data_dir)
        wrote = 0
        skipped = 0
        for image_path in images:
            out_path = image_path.with_suffix(image_path.suffix + ".prompt.txt")
            # Prefer cleaner sibling name: 0000.jpg -> 0000.prompt.txt
            out_path = image_path.with_name(image_path.stem + ".prompt.txt")
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            prompt, _ = resolve_prompt_for_image(image_path, index, append_style=append_style)
            out_path.write_text(prompt + "\n", encoding="utf-8")
            wrote += 1
        print(
            f"[generate_prompt] wrote={wrote} skipped={skipped} total_images={len(images)}",
            file=sys.stderr,
        )
        return 0

    print("[generate_prompt] provide --image_path or --data_dir", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Generate docs-ready Mintlify pages and copied assets from repository truth.

Sources:
- git history
- experiment CSV summaries
- figure metadata and saved figures
- core docs (README.md, ai_context.md, BUGFIXES.md)
- limited local archived Codex session traces when they mention this repo
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "portal"
GENERATED = PORTAL / "generated"
IMAGES = PORTAL / "images" / "generated"
FIGURES = ROOT / "figures"
EXPERIMENTS = ROOT / "experiments"
ARCHIVED_SESSIONS = Path.home() / ".codex" / "archived_sessions"
SESSION_INDEX = Path.home() / ".codex" / "session_index.jsonl"

SELECTED_FIGURES = [
    "reward_vs_training_steps.svg",
    "episode_length_vs_training_steps.svg",
    "crash_rate_vs_training_steps.svg",
    "curriculum_difficulty_vs_training_steps.svg",
    "ppo_policy_value_losses.svg",
    "hyperparameter_tuning_results.svg",
    "ablations_20260312_080245_reward_mean_std.svg",
    "ablations_20260312_080245_episode_length_mean_std.svg",
    "ablations_20260312_080245_crash_free_rate_mean_std.svg",
    "ablations_20260312_080245_curriculum_difficulty_mean_std.svg",
]

ABLATION_FIGURE_METADATA = {
    "ablations_20260312_080245_reward_mean_std.svg": {
        "caption": "Mean and standard deviation reward trajectories across the four ablation variants.",
        "description": "This figure compares reward progression across curriculum and non-curriculum variants, as well as single-environment and vectorized collection settings.",
        "section": "Ablation Study",
    },
    "ablations_20260312_080245_episode_length_mean_std.svg": {
        "caption": "Mean and standard deviation episode length trajectories across the four ablation variants.",
        "description": "This plot emphasizes survivability and control stability differences between the saved training variants.",
        "section": "Ablation Study",
    },
    "ablations_20260312_080245_crash_free_rate_mean_std.svg": {
        "caption": "Mean and standard deviation crash-free rate across the four ablation variants.",
        "description": "This view highlights the safety tradeoffs in the saved runs, especially when comparing curriculum progression against the stronger no-curriculum single-environment baseline.",
        "section": "Safety Analysis",
    },
    "ablations_20260312_080245_curriculum_difficulty_mean_std.svg": {
        "caption": "Mean and standard deviation curriculum difficulty progression across ablation variants.",
        "description": "Only curriculum-enabled variants climb the difficulty scale, making this plot the most direct visual check that the curriculum logic was actually engaged during those runs.",
        "section": "Method",
    },
}


@dataclass
class Highlight:
    label: str
    value: str
    evidence: str


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def choose_experiment_dir() -> Path | None:
    preferred = EXPERIMENTS / "20260311_230833"
    if preferred.exists():
        return preferred
    candidates = sorted([p for p in EXPERIMENTS.iterdir() if p.is_dir()])
    if not candidates:
        return None
    with_variant_summary = [p for p in candidates if (p / "variant_summary.csv").exists()]
    return with_variant_summary[-1] if with_variant_summary else candidates[-1]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_repo_docs() -> dict[str, str]:
    docs: dict[str, str] = {}
    for name in ["README.md", "ai_context.md", "BUGFIXES.md"]:
        path = ROOT / name
        if path.exists():
            docs[name] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def collect_git_history() -> list[dict[str, str]]:
    raw = run_git(
        "log",
        "--reverse",
        "--date=short",
        "--pretty=format:%H%x1f%ad%x1f%s%x1f%b%x1e",
    )
    commits: list[dict[str, str]] = []
    for block in raw.strip("\n\x1e").split("\x1e"):
        if not block.strip():
            continue
        commit_hash, date, subject, body = block.split("\x1f", 3)
        commits.append(
            {
                "hash": commit_hash.strip(),
                "short_hash": commit_hash.strip()[:7],
                "date": date.strip(),
                "subject": subject.strip(),
                "body": body.strip(),
            }
        )
    return commits


def files_for_commit(commit_hash: str) -> list[str]:
    raw = run_git("show", "--name-only", "--format=", commit_hash)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def clean_evidence_files(paths: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_parts = ("__pycache__", ".DS_Store", "events.out.tfevents", "checkpoints/latest.pth", "checkpoints/best.pth")
    for path in paths:
        if any(part in path for part in skip_parts):
            continue
        cleaned.append(path)
    preferred = [path for path in cleaned if not path.startswith(".minimax/")]
    return (preferred or cleaned)[:8]


def parse_bugfixes(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current_timestamp: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current_timestamp = line[3:].strip()
            continue
        if current_timestamp and line.startswith("- "):
            entries.append({"timestamp": current_timestamp, "detail": line[2:].strip()})
    return entries


def collect_session_mentions() -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    if not ARCHIVED_SESSIONS.exists():
        return mentions

    patterns = ("rl_neural_tesla", "car_neural_diff_learning", "Neural Tesla", "Autonomous Racing RL")
    for path in sorted(ARCHIVED_SESSIONS.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not any(pattern in text for pattern in patterns):
            continue
        mention_type = "repo_reference"
        if "inventory" in text.lower():
            mention_type = "inventory_reference"
        mentions.append(
            {
                "file": str(path),
                "date_hint": path.name.split("T")[0].replace("rollout-", ""),
                "type": mention_type,
            }
        )
    return mentions


def build_history_entries(commits: list[dict[str, str]], bugfixes: list[dict[str, str]]) -> list[dict[str, object]]:
    bugfix_lookup = "\n".join(f"{row['timestamp']}: {row['detail']}" for row in bugfixes)
    selected: list[dict[str, object]] = []

    for commit in commits:
        subject = commit["subject"].lower()
        short_hash = commit["short_hash"]
        files = clean_evidence_files(files_for_commit(commit["hash"]))
        entry: dict[str, object] | None = None

        if short_hash == "569f54c" or "initial project import" in subject:
            entry = {
                "title": "Initial RL system import",
                "date": commit["date"],
                "commit": short_hash,
                "problem": "The project needed a single repository that combined simulator code, policy/network code, PPO training code, tests, configs, and visualization hooks.",
                "change": "The initial import established the `rl_car_rl/` codebase and the basic documentation scaffold described in `README.md` and `ai_context.md`.",
                "outcome": "From this point on, the repo had a concrete baseline that could be trained, inspected, and iterated on instead of remaining a planning document.",
                "files": [
                    "README.md",
                    "ai_context.md",
                    "rl_car_rl/main.py",
                    "rl_car_rl/env/environment.py",
                    "rl_car_rl/env/track.py",
                    "rl_car_rl/agent/policy.py",
                    "rl_car_rl/training/trainer.py",
                    "rl_car_rl/tests/test_environment.py",
                ],
            }
        elif short_hash == "161f3d6" or ".minimax/skills" in subject:
            entry = {
                "title": "Repository cleanup for reviewability",
                "date": commit["date"],
                "commit": short_hash,
                "problem": "The repo included unrelated `.minimax/skills` content that distracted from the actual reinforcement learning project.",
                "change": "That folder was removed from version control so the top-level tree better reflected the project’s real scope.",
                "outcome": "The repository surface became easier for reviewers to scan, and the codebase looked more like a focused portfolio project instead of a mixed workspace dump.",
                "files": files,
            }
        elif short_hash == "8d79b82" or "device mismatch" in subject or "track rendering" in subject:
            entry = {
                "title": "Simulator and policy correctness fixes",
                "date": commit["date"],
                "commit": short_hash,
                "problem": "The training and rendering paths had multiple correctness issues: missing track dimension attributes, an MPS-versus-CPU action sampling mismatch, and rough track edges that produced visible corner artifacts.",
                "change": "The track now stores explicit pixel dimensions, the policy ensures the distribution parameters and sampled actions stay on the same device, and boundary generation smooths tangents to reduce kinks.",
                "outcome": "The project became more stable to run locally, more visually coherent, and less likely to fail during action sampling on Apple hardware. These fixes are corroborated by the detailed entries in `BUGFIXES.md`.",
                "files": files,
                "notes": bugfix_lookup,
            }
        elif short_hash == "6d9329f" or "curriculum update cadence" in subject:
            entry = {
                "title": "Curriculum progression logging fix",
                "date": commit["date"],
                "commit": short_hash,
                "problem": "The difficulty trace could remain flat because curriculum updates were tied to a counter that did not advance in the way the dashboard expected.",
                "change": "The training loop introduced a monotonic `global_step` and used it to drive curriculum update cadence more reliably.",
                "outcome": "Difficulty progression became observable in saved metrics and dashboards, which made the curriculum behavior auditable instead of opaque.",
                "files": files,
            }
        elif short_hash == "f3ecb80" or "ablation" in subject:
            entry = {
                "title": "Multi-seed ablation and figure packaging pass",
                "date": commit["date"],
                "commit": short_hash,
                "problem": "The repo needed stronger evidence than a single training run to support claims about curriculum learning, vectorized collection, and training behavior.",
                "change": "The experiment pipeline added multi-seed ablations across curriculum/no-curriculum and vectorized/single-environment variants, along with aggregate summaries, curriculum threshold reporting, and publication-style figures.",
                "outcome": "The repository gained reviewer-friendly evidence in `experiments/20260311_230833/` and the ablation SVGs under `figures/`, making it much easier to discuss trends with concrete numbers.",
                "files": [
                    "scripts/run_ablations.py",
                    "scripts/plot_ablations.py",
                    "experiments/20260311_230833/summary.csv",
                    "experiments/20260311_230833/variant_summary.csv",
                    "experiments/20260311_230833/curriculum_thresholds_summary.csv",
                    "figures/ablations_20260312_080245_reward_mean_std.svg",
                    "figures/ablations_20260312_080245_crash_free_rate_mean_std.svg",
                    "figures/figure_metadata.csv",
                ],
            }

        if entry is not None:
            selected.append(entry)

    return selected


def make_results_summary(experiment_dir: Path) -> tuple[list[Highlight], list[dict[str, str]]]:
    variant_rows = read_csv_rows(experiment_dir / "variant_summary.csv")
    summary_rows = read_csv_rows(experiment_dir / "summary.csv")
    threshold_rows = read_csv_rows(experiment_dir / "curriculum_thresholds_summary.csv")

    best_reward = max(variant_rows, key=lambda row: float(row["final_reward_mean_10_mean"]))
    best_crash_free = max(variant_rows, key=lambda row: float(row["final_crash_free_mean_10_mean"]))
    best_curriculum = max(variant_rows, key=lambda row: float(row["final_curriculum_level_mean"]))

    threshold_map = {row["variant"]: row for row in threshold_rows}
    curriculum_single = threshold_map.get("curriculum_single_env")
    curriculum_vectorized = threshold_map.get("curriculum_vectorized")

    highlights = [
        Highlight(
            label="Highest final reward mean",
            value=f"{best_reward['variant']} ({float(best_reward['final_reward_mean_10_mean']):.2f})",
            evidence="`experiments/20260311_230833/variant_summary.csv`",
        ),
        Highlight(
            label="Highest crash-free mean",
            value=f"{best_crash_free['variant']} ({float(best_crash_free['final_crash_free_mean_10_mean']):.2f})",
            evidence="`experiments/20260311_230833/variant_summary.csv`",
        ),
        Highlight(
            label="Farthest curriculum progression",
            value=f"{best_curriculum['variant']} (difficulty {float(best_curriculum['final_curriculum_level_mean']):.2f})",
            evidence="`experiments/20260311_230833/variant_summary.csv`",
        ),
    ]

    threshold_notes: list[dict[str, str]] = []
    if curriculum_single:
        threshold_notes.append(
            {
                "variant": "curriculum_single_env",
                "threshold_20": f"{float(curriculum_single['threshold_20_episode_mean']):.1f}",
                "threshold_40": f"{float(curriculum_single['threshold_40_episode_mean']):.1f}",
                "threshold_60": f"{float(curriculum_single['threshold_60_episode_mean']):.1f}",
                "threshold_80": f"{float(curriculum_single['threshold_80_episode_mean']):.1f}",
                "threshold_100": f"{float(curriculum_single['threshold_100_episode_mean']):.1f}",
            }
        )
    if curriculum_vectorized:
        threshold_notes.append(
            {
                "variant": "curriculum_vectorized",
                "threshold_20": f"{float(curriculum_vectorized['threshold_20_episode_mean']):.1f}",
                "threshold_40": f"{float(curriculum_vectorized['threshold_40_episode_mean']):.1f}",
                "threshold_60": f"{float(curriculum_vectorized['threshold_60_episode_mean']):.1f}",
                "threshold_80": "not reached",
                "threshold_100": "not reached",
            }
        )

    # Keep per-seed table for generated page context
    return highlights, summary_rows + threshold_notes


def copy_figures() -> list[Path]:
    copied: list[Path] = []
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name in SELECTED_FIGURES:
        src = FIGURES / name
        if not src.exists():
            continue
        dest = IMAGES / name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def load_figure_metadata() -> dict[str, dict[str, str]]:
    path = FIGURES / "figure_metadata.csv"
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    metadata = {row["filename"]: row for row in rows}
    metadata.update(ABLATION_FIGURE_METADATA)
    return metadata


def generate_repo_truth_page(repo_docs: dict[str, str], experiment_dir: Path | None, session_mentions: list[dict[str, str]]) -> str:
    experiment_note = str(experiment_dir.relative_to(ROOT)) if experiment_dir else "No experiment directory detected."
    session_line = (
        f"{len(session_mentions)} archived local Codex session file(s) mention the repo; current evidence is limited and used only as supplemental chronology."
        if session_mentions
        else "No archived local Codex session traces mentioning the repo were found."
    )
    return dedent(
        f"""\
        ---
        title: "Repo Truth Ledger"
        description: "The source material used to generate this portal, so reviewers can see what is narrative and what is derived."
        ---

        # Repo Truth Ledger

        This portal is built from repository evidence rather than from generic documentation filler. The generator script reads a narrow set of sources and turns them into reviewer-facing pages. That keeps the site honest about what is implemented, what is measured, and what is still incomplete.

        ## Core evidence sources

        - `README.md`: current high-level project framing and run instructions
        - `ai_context.md`: original scope and phased roadmap
        - `BUGFIXES.md`: explicit runtime fixes recorded with timestamps
        - `rl_car_rl/env/`, `rl_car_rl/agent/`, `rl_car_rl/training/`: implementation truth for simulator and learning system behavior
        - `{experiment_note}`: saved experiment summaries and per-run metrics
        - `figures/figure_metadata.csv` plus selected SVGs in `figures/`: captions and visual outputs
        - `git log`: repository chronology
        - archived local Codex session traces when available: supplemental context only

        ## Important constraint

        The portal does not pretend that deleted planning files such as `prd.md`, `tdd.md`, or `tasks.md` are still available. The current narrative uses the files that remain in the repository plus the actual commit history showing when those deletions happened.

        ## Session history note

        {session_line}

        ## Why this matters

        For a reviewer, the difference between a nice write-up and a trustworthy portal is traceability. This page exists so every strong claim on the site can be mapped back to code, data, logs, or version history already present in the repo.
        """
    )


def generate_build_history_page(entries: list[dict[str, object]], session_mentions: list[dict[str, str]]) -> str:
    sections: list[str] = []
    for entry in entries:
        files = entry.get("files", [])
        file_lines = "\n".join(f"- `{path}`" for path in files[:8]) or "- `git` metadata only"
        sections.append(
            "\n".join(
                [
                    f"## {entry['title']}",
                    "",
                    f"- Date: `{entry['date']}`",
                    f"- Commit: `{entry['commit']}`",
                    "",
                    "**Problem**",
                    "",
                    str(entry["problem"]),
                    "",
                    "**Change**",
                    "",
                    str(entry["change"]),
                    "",
                    "**Outcome**",
                    "",
                    str(entry["outcome"]),
                    "",
                    "**Primary evidence**",
                    "",
                    file_lines,
                ]
            )
        )

    session_note = (
        f"A supplemental archived session trace was found in `{Path(session_mentions[0]['file']).name}`. It confirms the repo name in a local project inventory, but it does not contain a detailed engineering chronology, so the build history still relies mainly on commits and saved artifacts."
        if session_mentions
        else "No archived local session trace with detailed engineering chronology was available, so this page is based primarily on commits, bug logs, and experiment timestamps."
    )

    body = "\n\n".join(sections)
    return "\n".join(
        [
            "---",
            'title: "Build History"',
            'description: "A major-change timeline organized around problem, change, and outcome rather than raw commit dumps."',
            "---",
            "",
            "# Build History",
            "",
            "This timeline selects the handful of repository changes that materially improved the project’s correctness, readability, or evidence quality. It intentionally skips low-signal maintenance commits and avoids presenting the git log as a substitute for explanation.",
            "",
            session_note,
            "",
            body,
        ]
    )


def generate_results_page(experiment_dir: Path, highlights: list[Highlight], summary_rows: list[dict[str, str]]) -> str:
    top_table = "\n".join(
        f"| {highlight.label} | {highlight.value} | {highlight.evidence} |"
        for highlight in highlights
    )

    variant_rows = read_csv_rows(experiment_dir / "variant_summary.csv")
    variant_table = "\n".join(
        "| {variant} | {reward:.2f} | {length:.2f} | {crash_free:.2f} | {difficulty:.2f} |".format(
            variant=row["variant"],
            reward=float(row["final_reward_mean_10_mean"]),
            length=float(row["final_length_mean_10_mean"]),
            crash_free=float(row["final_crash_free_mean_10_mean"]),
            difficulty=float(row["final_curriculum_level_mean"]),
        )
        for row in variant_rows
    )

    threshold_rows = read_csv_rows(experiment_dir / "curriculum_thresholds_summary.csv")
    threshold_table = "\n".join(
        "| {variant} | {t20} | {t40} | {t60} | {t80} | {t100} |".format(
            variant=row["variant"],
            t20=(f"{float(row['threshold_20_episode_mean']):.1f}" if row["threshold_20_episode_mean"] else "not reached"),
            t40=(f"{float(row['threshold_40_episode_mean']):.1f}" if row["threshold_40_episode_mean"] else "not reached"),
            t60=(f"{float(row['threshold_60_episode_mean']):.1f}" if row["threshold_60_episode_mean"] else "not reached"),
            t80=(f"{float(row['threshold_80_episode_mean']):.1f}" if row["threshold_80_episode_mean"] else "not reached"),
            t100=(f"{float(row['threshold_100_episode_mean']):.1f}" if row["threshold_100_episode_mean"] else "not reached"),
        )
        for row in threshold_rows
    )

    return "\n".join(
        [
            "---",
            'title: "Results Highlights"',
            'description: "High-signal findings extracted from the saved ablation summaries and curriculum threshold reports."',
            "---",
            "",
            "# Results Highlights",
            "",
            f"The strongest experiment evidence currently in the repo comes from `{experiment_dir.relative_to(ROOT)}`. That directory contains four variants, three seeds per variant, aggregate curves, per-seed summaries, and curriculum threshold summaries. The goal of this page is to surface the strongest claims from that bundle without turning the page into a raw file listing.",
            "",
            "## Snapshot highlights",
            "",
            "| Highlight | Value | Evidence |",
            "| --- | --- | --- |",
            top_table,
            "",
            "## Variant summary",
            "",
            "| Variant | Final reward mean | Final length mean | Final crash-free mean | Final curriculum level mean |",
            "| --- | --- | --- | --- | --- |",
            variant_table,
            "",
            "A few patterns stand out immediately. The single-environment, no-curriculum variant ends with the highest mean reward and the strongest crash-free rate in the saved summary. The curriculum-enabled single-environment variant reaches the full difficulty ceiling of `1.0`, while the curriculum-enabled vectorized variant progresses partway through the scale but not to the final threshold in the saved 1000-episode runs.",
            "",
            "## Curriculum thresholds",
            "",
            "| Variant | Threshold 0.2 | Threshold 0.4 | Threshold 0.6 | Threshold 0.8 | Threshold 1.0 |",
            "| --- | --- | --- | --- | --- | --- |",
            threshold_table,
            "",
            "The threshold table makes the tradeoff visible: `curriculum_single_env` climbs the difficulty ladder much faster in the saved runs, while `curriculum_vectorized` reaches intermediate thresholds later on average and does not reach `0.8` or `1.0` within the reported horizon.",
            "",
            "## Scope note",
            "",
            "These findings are intentionally narrow. They describe the saved experiment bundle that exists in the repo today. They do not claim broader generality than the current runs support.",
        ]
    )


def generate_gallery_page(metadata: dict[str, dict[str, str]], copied_figures: list[Path]) -> str:
    blocks: list[str] = []
    for image_path in copied_figures:
        name = image_path.name
        meta = metadata.get(name) or metadata.get(name.replace("_v2", "")) or {
            "caption": name,
            "description": "Saved repository artifact.",
            "section": "Artifacts",
        }
        blocks.append(
            dedent(
                f"""\
                ## {name}

                ![{name}](/images/generated/{name})

                **Caption:** {meta['caption']}

                **Why it matters:** {meta['description']}

                **Original section tag:** {meta['section']}
                """
            ).strip()
        )

    body = "\n\n".join(blocks)
    return "\n".join(
        [
            "---",
            'title: "Generated Artifact Gallery"',
            'description: "Selected training and ablation visuals copied into the Mintlify portal."',
            "---",
            "",
            "# Generated Artifact Gallery",
            "",
            "This page is produced from the current `figures/` directory and the caption metadata stored in `figures/figure_metadata.csv`. Only a curated subset of assets is included so the portal remains reviewer-friendly.",
            "",
            body,
        ]
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def parse_docs_json_nav_pages(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages: list[str] = []
    for group in data.get("navigation", []):
        pages.extend(group.get("pages", []))
    return pages


def validate_portal() -> tuple[bool, list[str]]:
    errors: list[str] = []
    docs_json = PORTAL / "docs.json"
    if not docs_json.exists():
        errors.append("Missing portal/docs.json")
        return False, errors

    try:
        nav_pages = parse_docs_json_nav_pages(docs_json)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"docs.json parse failed: {exc}")
        return False, errors

    for page in nav_pages:
        mdx_path = PORTAL / f"{page}.mdx"
        if not mdx_path.exists():
            errors.append(f"Navigation target missing: {mdx_path}")

    image_pattern = re.compile(r"!\[[^\]]*\]\((/images/generated/[^)]+)\)")
    for mdx_path in PORTAL.rglob("*.mdx"):
        text = mdx_path.read_text(encoding="utf-8", errors="ignore")
        for image_ref in image_pattern.findall(text):
            image_path = PORTAL / image_ref.lstrip("/")
            if not image_path.exists():
                errors.append(f"Image reference missing: {image_path} (from {mdx_path})")

    return not errors, errors


def generate() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    repo_docs = load_repo_docs()
    commits = collect_git_history()
    bugfixes = parse_bugfixes(repo_docs.get("BUGFIXES.md", ""))
    session_mentions = collect_session_mentions()
    history_entries = build_history_entries(commits, bugfixes)
    figure_metadata = load_figure_metadata()
    copied_figures = copy_figures()

    experiment_dir = choose_experiment_dir()
    if experiment_dir is None:
        raise RuntimeError("No experiment directory found under experiments/")

    highlights, _ = make_results_summary(experiment_dir)

    write_text(
        GENERATED / "repo-truth.mdx",
        generate_repo_truth_page(repo_docs, experiment_dir, session_mentions),
    )
    write_text(
        GENERATED / "build-history.mdx",
        generate_build_history_page(history_entries, session_mentions),
    )
    write_text(
        GENERATED / "results-highlights.mdx",
        generate_results_page(experiment_dir, highlights, []),
    )
    write_text(
        GENERATED / "artifact-gallery.mdx",
        generate_gallery_page(figure_metadata, copied_figures),
    )

    manifest = {
        "experiment_dir": str(experiment_dir),
        "copied_figures": [str(path.relative_to(ROOT)) for path in copied_figures],
        "history_entries": len(history_entries),
        "session_mentions": len(session_mentions),
    }
    write_text(GENERATED / "portal_manifest.json", json.dumps(manifest, indent=2))


def main() -> None:
    validate_only = "--validate-only" in os.sys.argv
    if not validate_only:
        generate()

    ok, errors = validate_portal()
    if not ok:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Portal generation/validation passed.")


if __name__ == "__main__":
    main()

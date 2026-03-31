import json
from pathlib import Path


TRACK_KEYS = (
    "timbre_global_cos_to_timbre_ref",
    "global_cos_to_style_ref",
    "prosody_cos_to_style_ref",
    "dynamic_timbre_cos_to_dynamic_timbre_ref",
    "ext_speaker_cos_to_timbre_ref",
    "ext_content_ssl_cos_to_src",
)

TRACK_LABELS = {
    "avg_timbre_global_cos_to_timbre_ref": "Timbre->TimbreRef",
    "avg_global_cos_to_style_ref": "Style->StyleRef",
    "avg_prosody_cos_to_style_ref": "Prosody->StyleRef",
    "avg_dynamic_timbre_cos_to_dynamic_timbre_ref": "Dynamic->DynamicRef",
    "avg_ext_speaker_cos_to_timbre_ref": "ExtSpeaker->TimbreRef",
    "avg_ext_content_ssl_cos_to_src": "ExtContent->Src",
}


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values, default=None):
    values = [float(v) for v in values if v is not None]
    if not values:
        return default
    return float(sum(values) / len(values))


def _metric_header_suffix():
    return " | ".join(
        [
            TRACK_LABELS["avg_timbre_global_cos_to_timbre_ref"],
            TRACK_LABELS["avg_global_cos_to_style_ref"],
            TRACK_LABELS["avg_prosody_cos_to_style_ref"],
            TRACK_LABELS["avg_dynamic_timbre_cos_to_dynamic_timbre_ref"],
            TRACK_LABELS["avg_ext_speaker_cos_to_timbre_ref"],
            TRACK_LABELS["avg_ext_content_ssl_cos_to_src"],
        ]
    )


def _metric_separator_suffix():
    return " | ".join(["---:"] * 6)


def load_factorized_evaluation(sweep_dir):
    sweep_dir = Path(sweep_dir)
    eval_path = sweep_dir / "evaluation_summary.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {eval_path}")
    with open(eval_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = [row for row in rows if row.get("swap_variant") is not None]
    if not rows:
        raise ValueError("No factorized swap rows found in evaluation summary.")
    return sweep_dir, rows


def build_swap_variant_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get("swap_variant")), []).append(row)

    summary = []
    for variant, items in sorted(groups.items()):
        entry = {
            "swap_variant": variant,
            "num_rows": len(items),
        }
        for key in TRACK_KEYS:
            entry[f"avg_{key}"] = _mean([_safe_float(item.get(key)) for item in items], default=None)
        summary.append(entry)
    return summary


def build_profile_variant_summary(rows):
    groups = {}
    for row in rows:
        profile = str(row.get("style_profile") or row.get("profile") or "unknown")
        variant = str(row.get("swap_variant") or "unknown")
        groups.setdefault((profile, variant), []).append(row)

    summary = []
    for (profile, variant), items in sorted(groups.items()):
        entry = {
            "profile": profile,
            "swap_variant": variant,
            "num_rows": len(items),
        }
        for key in TRACK_KEYS:
            entry[f"avg_{key}"] = _mean([_safe_float(item.get(key)) for item in items], default=None)
        summary.append(entry)
    return summary


def build_profile_summary(rows):
    groups = {}
    for row in rows:
        profile = str(row.get("style_profile") or row.get("profile") or "unknown")
        groups.setdefault(profile, []).append(row)

    summary = []
    for profile, items in sorted(groups.items()):
        entry = {
            "profile": profile,
            "num_rows": len(items),
        }
        for key in TRACK_KEYS:
            entry[f"avg_{key}"] = _mean([_safe_float(item.get(key)) for item in items], default=None)
        summary.append(entry)
    return summary


def render_markdown(summary, sweep_dir):
    header_suffix = _metric_header_suffix()
    separator_suffix = _metric_separator_suffix()
    lines = [
        "# Factorized Swap Matrix Report",
        "",
        f"- sweep_dir: `{sweep_dir}`",
        "",
        "## Profile x Variant",
        "",
        f"| Profile | Variant | Rows | {header_suffix} |",
        f"| --- | --- | ---: | {separator_suffix} |",
    ]
    for item in summary["by_profile_variant"]:
        lines.append(
            "| {profile} | {swap_variant} | {num_rows} | {timbre} | {style} | {prosody} | {dynamic} | {ext_spk} | {ext_ctc} |".format(
                profile=item["profile"],
                swap_variant=item["swap_variant"],
                num_rows=item["num_rows"],
                timbre=_fmt(item.get("avg_timbre_global_cos_to_timbre_ref")),
                style=_fmt(item.get("avg_global_cos_to_style_ref")),
                prosody=_fmt(item.get("avg_prosody_cos_to_style_ref")),
                dynamic=_fmt(item.get("avg_dynamic_timbre_cos_to_dynamic_timbre_ref")),
                ext_spk=_fmt(item.get("avg_ext_speaker_cos_to_timbre_ref")),
                ext_ctc=_fmt(item.get("avg_ext_content_ssl_cos_to_src")),
            )
        )

    lines.extend(
        [
            "",
            "## Profile Aggregate",
            "",
            f"| Profile | Rows | {header_suffix} |",
            f"| --- | ---: | {separator_suffix} |",
        ]
    )
    for item in summary["by_profile"]:
        lines.append(
            "| {profile} | {num_rows} | {timbre} | {style} | {prosody} | {dynamic} | {ext_spk} | {ext_ctc} |".format(
                profile=item["profile"],
                num_rows=item["num_rows"],
                timbre=_fmt(item.get("avg_timbre_global_cos_to_timbre_ref")),
                style=_fmt(item.get("avg_global_cos_to_style_ref")),
                prosody=_fmt(item.get("avg_prosody_cos_to_style_ref")),
                dynamic=_fmt(item.get("avg_dynamic_timbre_cos_to_dynamic_timbre_ref")),
                ext_spk=_fmt(item.get("avg_ext_speaker_cos_to_timbre_ref")),
                ext_ctc=_fmt(item.get("avg_ext_content_ssl_cos_to_src")),
            )
        )

    lines.extend(
        [
            "",
            "## Variant Aggregate",
            "",
            f"| Variant | Rows | {header_suffix} |",
            f"| --- | ---: | {separator_suffix} |",
        ]
    )
    for item in summary["by_variant"]:
        lines.append(
            "| {swap_variant} | {num_rows} | {timbre} | {style} | {prosody} | {dynamic} | {ext_spk} | {ext_ctc} |".format(
                swap_variant=item["swap_variant"],
                num_rows=item["num_rows"],
                timbre=_fmt(item.get("avg_timbre_global_cos_to_timbre_ref")),
                style=_fmt(item.get("avg_global_cos_to_style_ref")),
                prosody=_fmt(item.get("avg_prosody_cos_to_style_ref")),
                dynamic=_fmt(item.get("avg_dynamic_timbre_cos_to_dynamic_timbre_ref")),
                ext_spk=_fmt(item.get("avg_ext_speaker_cos_to_timbre_ref")),
                ext_ctc=_fmt(item.get("avg_ext_content_ssl_cos_to_src")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def write_factorized_swap_report(sweep_dir):
    sweep_dir, rows = load_factorized_evaluation(sweep_dir)
    summary = {
        "by_variant": build_swap_variant_summary(rows),
        "by_profile": build_profile_summary(rows),
        "by_profile_variant": build_profile_variant_summary(rows),
    }
    json_path = sweep_dir / "factorized_swap_summary.json"
    md_path = sweep_dir / "factorized_swap_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(summary, sweep_dir))
    return {
        "sweep_dir": str(sweep_dir),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "num_variants": len(summary["by_variant"]),
        "num_profiles": len(summary["by_profile"]),
        "num_profile_variant_rows": len(summary["by_profile_variant"]),
    }

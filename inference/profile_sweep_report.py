import json
import math
from pathlib import Path


DISTANCE_TO_REF_KEYS = (
    "mel_mean_l1_to_ref",
    "mel_std_l1_to_ref",
    "energy_mean_abs_to_ref",
    "energy_std_abs_to_ref",
    "f0_mean_abs_to_ref",
    "f0_std_abs_to_ref",
    "voiced_frac_abs_to_ref",
)

COSINE_TO_REF_KEYS = (
    "global_cos_to_ref",
    "timbre_global_cos_to_ref",
    "prosody_cos_to_ref",
    "dynamic_timbre_cos_to_ref",
)

SCORABLE_DISTANCE_PREFIXES = (
    "mel_mean_l1_to_",
    "mel_std_l1_to_",
    "energy_mean_abs_to_",
    "energy_std_abs_to_",
    "f0_mean_abs_to_",
    "f0_std_abs_to_",
    "voiced_frac_abs_to_",
)

SCORABLE_COSINE_PREFIXES = (
    "global_cos_to_",
    "timbre_global_cos_to_",
    "prosody_cos_to_",
    "dynamic_timbre_cos_to_",
)


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _mean(values, default=0.0):
    filtered = [float(v) for v in values if v is not None]
    if len(filtered) <= 0:
        return float(default)
    return float(sum(filtered) / len(filtered))


def _collect_scorable_metric_keys(rows, explicit_keys, prefixes):
    keys = list(explicit_keys)
    seen = set(keys)
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key in seen:
                continue
            if any(str(key).startswith(prefix) for prefix in prefixes):
                keys.append(key)
                seen.add(key)
    return tuple(keys)


def load_report_inputs(sweep_dir):
    sweep_dir = Path(sweep_dir)
    eval_path = sweep_dir / "evaluation_summary.json"
    manifest_path = sweep_dir / "sweep_manifest.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {eval_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing sweep manifest: {manifest_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        evaluation_rows = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = json.load(f)

    manifest_index = {
        (row.get("case_name"), row.get("profile")): row
        for row in manifest_rows
        if row.get("status") == "ok"
    }
    return sweep_dir, evaluation_rows, manifest_index


def _load_meta(sweep_dir, manifest_row):
    meta_path = sweep_dir / manifest_row["meta_path"]
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_results_with_meta(sweep_dir, evaluation_rows, manifest_index):
    enriched = []
    for row in evaluation_rows:
        row = dict(row)
        manifest_row = manifest_index.get((row.get("case_name"), row.get("profile")))
        if manifest_row is None:
            enriched.append(row)
            continue
        meta = _load_meta(sweep_dir, manifest_row)
        row["resolved_profile"] = meta.get("resolved_profile", {})
        row["model_input"] = meta.get("model_input", {})
        row["meta_path"] = manifest_row.get("meta_path")
        row["mel_path"] = manifest_row.get("mel_path")
        enriched.append(row)
    return enriched


def _normalize_case_metric(rows, key, *, higher_is_better):
    values = [_safe_float(row.get(key), default=None) for row in rows]
    valid = [v for v in values if v is not None]
    if len(valid) <= 0:
        return {id(row): None for row in rows}
    vmin = min(valid)
    vmax = max(valid)
    if abs(vmax - vmin) < 1e-8:
        return {id(row): 1.0 for row in rows if _safe_float(row.get(key), default=None) is not None}

    normalized = {}
    for row in rows:
        value = _safe_float(row.get(key), default=None)
        if value is None:
            normalized[id(row)] = None
            continue
        score = (value - vmin) / (vmax - vmin)
        normalized[id(row)] = float(score if higher_is_better else 1.0 - score)
    return normalized


def score_results_by_case(rows):
    distance_metric_keys = _collect_scorable_metric_keys(
        rows,
        DISTANCE_TO_REF_KEYS,
        SCORABLE_DISTANCE_PREFIXES,
    )
    cosine_metric_keys = _collect_scorable_metric_keys(
        rows,
        COSINE_TO_REF_KEYS,
        SCORABLE_COSINE_PREFIXES,
    )
    rows_by_case = {}
    for row in rows:
        rows_by_case.setdefault(row.get("case_name", "unknown"), []).append(row)

    for case_rows in rows_by_case.values():
        ref_scores = {id(row): [] for row in case_rows}
        stability_scores = {id(row): [] for row in case_rows}

        for key in distance_metric_keys:
            normalized = _normalize_case_metric(case_rows, key, higher_is_better=False)
            for row in case_rows:
                value = normalized.get(id(row))
                if value is not None:
                    ref_scores[id(row)].append(value)

        for key in cosine_metric_keys:
            normalized = _normalize_case_metric(case_rows, key, higher_is_better=True)
            for row in case_rows:
                value = normalized.get(id(row))
                if value is not None:
                    ref_scores[id(row)].append(value)

        for row in case_rows:
            duration_ratio = _safe_float(row.get("duration_ratio_to_src"), default=None)
            if duration_ratio is not None:
                stability_scores[id(row)].append(max(0.0, 1.0 - abs(duration_ratio - 1.0)))
            elapsed = _safe_float(row.get("elapsed_sec"), default=None)
            if elapsed is not None:
                stability_scores[id(row)].append(1.0 / (1.0 + max(elapsed, 0.0)))

        for row in case_rows:
            row["ref_similarity_score"] = round(_mean(ref_scores[id(row)], default=0.0), 6)
            row["stability_score"] = round(_mean(stability_scores[id(row)], default=0.0), 6)
            row["overall_score"] = round(
                0.8 * row["ref_similarity_score"] + 0.2 * row["stability_score"],
                6,
            )

        ranked = sorted(case_rows, key=lambda item: item.get("overall_score", 0.0), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["case_rank"] = rank
            row["case_best"] = rank == 1
    return rows


def aggregate_profile_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.get("profile", "unknown"), []).append(row)

    summary = []
    for profile, items in sorted(groups.items()):
        summary.append(
            {
                "profile": profile,
                "num_cases": len(items),
                "avg_overall_score": round(_mean([item.get("overall_score") for item in items]), 6),
                "avg_ref_similarity_score": round(_mean([item.get("ref_similarity_score") for item in items]), 6),
                "avg_stability_score": round(_mean([item.get("stability_score") for item in items]), 6),
                "num_case_wins": int(sum(1 for item in items if item.get("case_best"))),
                "avg_elapsed_sec": round(_mean([item.get("elapsed_sec") for item in items]), 6),
            }
        )
    summary.sort(key=lambda item: item["avg_overall_score"], reverse=True)
    return summary


def build_case_summary(rows):
    case_groups = {}
    for row in rows:
        case_groups.setdefault(row.get("case_name", "unknown"), []).append(row)
    for items in case_groups.values():
        items.sort(key=lambda item: item.get("overall_score", 0.0), reverse=True)
    return case_groups


def _render_profile_summary_markdown(profile_summary):
    lines = [
        "## Profile Summary",
        "",
        "| Profile | Cases | Avg Overall | Avg Ref Similarity | Avg Stability | Wins | Avg Time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in profile_summary:
        lines.append(
            f"| {item['profile']} | {item['num_cases']} | {item['avg_overall_score']:.4f} | "
            f"{item['avg_ref_similarity_score']:.4f} | {item['avg_stability_score']:.4f} | "
            f"{item['num_case_wins']} | {item['avg_elapsed_sec']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_case_markdown(sweep_dir, case_name, rows):
    lines = [f"## {case_name}", ""]
    for row in rows:
        wav_path = row.get("wav_path", "")
        wav_link = f"../{wav_path}" if wav_path else ""
        lines.extend(
            [
                f"### {row.get('profile')} (rank #{row.get('case_rank')})",
                f"- overall_score: {row.get('overall_score', 0.0):.4f}",
                f"- ref_similarity_score: {row.get('ref_similarity_score', 0.0):.4f}",
                f"- stability_score: {row.get('stability_score', 0.0):.4f}",
                f"- wav: [{wav_path}]({wav_link})" if wav_path else "- wav: N/A",
                "",
            ]
        )
    return "\n".join(lines)


def render_markdown_report(sweep_dir, profile_summary, case_summary):
    lines = [
        "# Style Profile Sweep Report",
        "",
        f"- sweep_dir: `{sweep_dir}`",
        "",
        _render_profile_summary_markdown(profile_summary),
    ]
    for case_name, rows in case_summary.items():
        lines.append(_render_case_markdown(sweep_dir, case_name, rows))
    return "\n".join(lines) + "\n"


def _html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html_report(sweep_dir, profile_summary, case_summary):
    profile_rows = "\n".join(
        [
            "<tr>"
            f"<td>{_html_escape(item['profile'])}</td>"
            f"<td>{item['num_cases']}</td>"
            f"<td>{item['avg_overall_score']:.4f}</td>"
            f"<td>{item['avg_ref_similarity_score']:.4f}</td>"
            f"<td>{item['avg_stability_score']:.4f}</td>"
            f"<td>{item['num_case_wins']}</td>"
            f"<td>{item['avg_elapsed_sec']:.4f}</td>"
            "</tr>"
            for item in profile_summary
        ]
    )

    case_sections = []
    for case_name, rows in case_summary.items():
        profile_cards = []
        for row in rows:
            wav_path = row.get("wav_path", "")
            wav_src = _html_escape(f"../{wav_path}" if wav_path else "")
            resolved_profile = row.get("resolved_profile", {})
            profile_cards.append(
                f"""
                <div class="card {'best' if row.get('case_best') else ''}">
                  <h3>{_html_escape(row.get('profile'))} <span class="rank">#{row.get('case_rank')}</span></h3>
                  <p>overall: <strong>{row.get('overall_score', 0.0):.4f}</strong></p>
                  <p>ref similarity: {row.get('ref_similarity_score', 0.0):.4f}</p>
                  <p>stability: {row.get('stability_score', 0.0):.4f}</p>
                  <audio controls preload="none" src="{wav_src}"></audio>
                  <details>
                    <summary>resolved profile params</summary>
                    <pre>{_html_escape(json.dumps(resolved_profile, ensure_ascii=False, indent=2))}</pre>
                  </details>
                </div>
                """
            )
        case_sections.append(
            f"""
            <section class="case-block">
              <h2>{_html_escape(case_name)}</h2>
              <div class="card-grid">
                {''.join(profile_cards)}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Style Profile Sweep Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .case-block {{ margin-top: 28px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 14px; background: #fff; }}
    .card.best {{ border-color: #2a7; box-shadow: 0 0 0 2px rgba(34,170,119,0.12); }}
    .rank {{ color: #666; font-size: 0.9em; }}
    audio {{ width: 100%; margin: 10px 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 10px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Style Profile Sweep Report</h1>
  <p><strong>sweep_dir:</strong> {_html_escape(sweep_dir)}</p>
  <h2>Profile Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Profile</th><th>Cases</th><th>Avg Overall</th><th>Avg Ref Similarity</th>
        <th>Avg Stability</th><th>Wins</th><th>Avg Time (s)</th>
      </tr>
    </thead>
    <tbody>{profile_rows}</tbody>
  </table>
  {''.join(case_sections)}
</body>
</html>
"""


class StyleProfileSweepReporter:
    def __init__(self, sweep_dir):
        self.sweep_dir = Path(sweep_dir)

    def build(self):
        sweep_dir, evaluation_rows, manifest_index = load_report_inputs(self.sweep_dir)
        rows = enrich_results_with_meta(sweep_dir, evaluation_rows, manifest_index)
        rows = score_results_by_case(rows)
        profile_summary = aggregate_profile_summary(rows)
        case_summary = build_case_summary(rows)
        return rows, profile_summary, case_summary

    def write_reports(self):
        rows, profile_summary, case_summary = self.build()
        report_dir = self.sweep_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        enriched_json_path = report_dir / "report_data.json"
        with open(enriched_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rows": rows,
                    "profile_summary": profile_summary,
                    "case_summary": case_summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        markdown = render_markdown_report(self.sweep_dir, profile_summary, case_summary)
        markdown_path = report_dir / "report.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        html = render_html_report(self.sweep_dir, profile_summary, case_summary)
        html_path = report_dir / "report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        report = {
            "report_dir": str(report_dir),
            "report_md": str(markdown_path),
            "report_html": str(html_path),
            "report_data": str(enriched_json_path),
            "num_rows": len(rows),
            "num_profiles": len(profile_summary),
            "num_cases": len(case_summary),
        }
        with open(report_dir / "report_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

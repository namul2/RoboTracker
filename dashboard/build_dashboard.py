#!/usr/bin/env python3
"""RoboTracker dashboard generator.

Scans the local ``outputs/`` directory produced by ``visualize.py``,
``image_clustering.py``, ``ood_detection.py`` and ``ood_episode_inspector.py``
and renders a single self-contained ``dashboard/index.html``.

The JSON summaries store *absolute* paths from the machine that produced them
(e.g. ``/hdd_1/...``), so we never trust those paths. Instead we walk the real
``outputs/`` tree on disk and link images with paths relative to this folder.

Usage:
    python3 dashboard/build_dashboard.py
    # then open dashboard/index.html  (or: python3 -m http.server -d . )
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUTS = os.path.join(ROOT, "outputs")

# Directories under outputs/ we never surface in the dashboard.
EXCLUDE_DIRS = {"tmp_old", "first_frames"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def rel(path):
    """Path relative to the dashboard directory, web-safe."""
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def exists(path):
    return path and os.path.isfile(path)


def fmt(v, nd=3):
    """Human-friendly number formatting."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v == 0:
            return "0"
        av = abs(v)
        if av < 1e-3 or av >= 1e5:
            return f"{v:.2e}"
        return f"{round(v, nd):,}".rstrip("0").rstrip(".") if "." in f"{round(v, nd)}" else f"{v:,}"
    return str(v)


def esc(s):
    return html.escape(str(s))


def vec(values, nd=3):
    if not values:
        return "–"
    return "[" + ", ".join(fmt(v, nd) for v in values) + "]"


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def collect_trajectory():
    base = os.path.join(OUTPUTS, "trajectory_distribution")
    if not os.path.isdir(base):
        return []
    datasets = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d) or name in EXCLUDE_DIRS:
            continue
        summary = load_json(os.path.join(d, "summary.json"))
        if summary is None:
            continue
        is_bi = bool(summary.get("has_fk")) and "left" in summary and "right" in summary

        def imgs(*names, sub=""):
            out = []
            for n in names:
                p = os.path.join(d, sub, n) if sub else os.path.join(d, n)
                if exists(p):
                    out.append({"src": rel(p), "label": n.replace("_", " ").rsplit(".", 1)[0]})
            return out

        entry = {"name": name, "is_bimanual": is_bi, "summary": summary}
        if is_bi:
            entry["combined"] = imgs("ee_bimanual_combined_3d.png")
            entry["left"] = imgs(
                "ee_position_3d.png", "ee_projection_xy.png", "ee_projection_yz.png",
                "ee_projection_xz.png", "ee_speed_hist.png", "joint_distributions.png", sub="left",
            )
            entry["right"] = imgs(
                "ee_position_3d.png", "ee_projection_xy.png", "ee_projection_yz.png",
                "ee_projection_xz.png", "ee_speed_hist.png", "joint_distributions.png", sub="right",
            )
        else:
            entry["images"] = imgs(
                "position_distribution_3d.png", "position_distribution_2d.png",
                "position_projection_xy.png", "position_projection_yz.png",
                "position_projection_xz.png", "ee_speed_distribution_hist.png",
            )
        datasets.append(entry)
    return datasets


def collect_image_distribution():
    base = os.path.join(OUTPUTS, "image_distribution")
    if not os.path.isdir(base):
        return None
    summary = load_json(os.path.join(base, "summary.json")) or {}

    def imgs(folder, names):
        out = []
        for n in names:
            p = os.path.join(folder, n)
            if exists(p):
                out.append({"src": rel(p), "label": n.replace("_", " ").rsplit(".", 1)[0]})
        return out

    global_imgs = imgs(base, [
        "embedding_clusters.png", "embedding_datasets.png",
        "embedding_cameras.png", "cluster_contact_sheet.png",
    ])

    per = []
    for name, meta in (summary.get("datasets") or {}).items():
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        per.append({
            "name": name,
            "meta": meta,
            "embedding": (summary.get("per_dataset_embeddings") or {}).get(name, {}),
            "images": imgs(d, [
                "embedding_clusters.png", "embedding_cameras.png",
                "cluster_contact_sheet.png", "camera_pose_map_3d.png",
            ]),
        })
    return {"summary": summary, "global_images": global_imgs, "per_dataset": per}


def _find_ood_runs(base):
    """Locate every directory containing an ood_summary.json under base."""
    runs = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [x for x in dirnames if x not in EXCLUDE_DIRS]
        if "ood_summary.json" in filenames:
            runs.append(dirpath)
    return sorted(runs)


def collect_ood_detection():
    base = os.path.join(OUTPUTS, "ood_detection")
    if not os.path.isdir(base):
        return []
    runs = []
    for d in _find_ood_runs(base):
        summary = load_json(os.path.join(d, "ood_summary.json"))
        if summary is None:
            continue

        def imgs(names):
            out = []
            for n in names:
                p = os.path.join(d, n)
                if exists(p):
                    out.append({"src": rel(p), "label": n.replace("_", " ").rsplit(".", 1)[0]})
            return out

        report = read_text(os.path.join(d, "ood_report.txt"))
        runs.append({
            "name": summary.get("name", os.path.basename(d)),
            "path": rel(d),
            "summary": summary,
            "report": report,
            "images": imgs([
                "ood_signal_bar.png", "trajectory_train_test_3d.png",
                "speed_train_test.png", "initial_environment_3d.png",
                "image_embedding_train_test.png", "mean_comparison_bar.png",
            ]),
        })
    return runs


def collect_episode_inspector():
    base = os.path.join(OUTPUTS, "ood_episode_inspector")
    if not os.path.isdir(base):
        return []
    runs = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [x for x in dirnames if x not in EXCLUDE_DIRS]
        if "episode_inspector_summary.json" not in filenames:
            continue
        summary = load_json(os.path.join(dirpath, "episode_inspector_summary.json"))
        if summary is None:
            continue

        def imgs(names):
            out = []
            for n in names:
                p = os.path.join(dirpath, n)
                if exists(p):
                    out.append({"src": rel(p), "label": n.replace("_", " ").rsplit(".", 1)[0]})
            return out

        runs.append({
            "name": summary.get("name", os.path.basename(dirpath)),
            "summary": summary,
            "images": imgs([
                "ranked_episode_scores.png", "score_distribution.png",
                "reason_signal_heatmap.png", "motion_metric_comparison.png",
            ]),
        })
    return sorted(runs, key=lambda r: r["name"])


# --------------------------------------------------------------------------- #
# HTML rendering pieces
# --------------------------------------------------------------------------- #
def img_grid(images):
    if not images:
        return ""
    cells = "".join(
        f'<figure class="shot" onclick="zoom(this)">'
        f'<img loading="lazy" src="{esc(im["src"])}" alt="{esc(im["label"])}">'
        f'<figcaption>{esc(im["label"])}</figcaption></figure>'
        for im in images
    )
    return f'<div class="grid">{cells}</div>'


def stat_chips(pairs):
    chips = "".join(
        f'<div class="chip"><span class="k">{esc(k)}</span>'
        f'<span class="v">{v}</span></div>'
        for k, v in pairs
    )
    return f'<div class="chips">{chips}</div>'


def signal_bars(metrics):
    """OOD shift metrics rendered as labeled bars (0-0.5 ok, 0.5-1.5 mod, 1.5+ strong)."""
    rows = []
    for k, v in metrics.items():
        if v is None:
            continue
        val = float(v)
        pct = max(2.0, min(val / 2.5 * 100.0, 100.0))
        cls = "ok" if val < 0.5 else ("mod" if val < 1.5 else "strong")
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{esc(k)}</div>'
            f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="bar-val">{fmt(val)}</div></div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'


def speed_block(title, s, unit=""):
    pairs = [
        ("frames", fmt(s.get("frame_count") or s.get("speed_count"))),
        ("speed mean", fmt(s.get("speed_mean"))),
        ("speed median", fmt(s.get("speed_median"))),
        ("speed p95", fmt(s.get("speed_p95"))),
        ("speed max", fmt(s.get("speed_max"))),
        ("pos min", vec(s.get("position_min"))),
        ("pos max", vec(s.get("position_max"))),
        ("pos mean", vec(s.get("position_mean"))),
        ("grip-change eps", fmt(s.get("episodes_with_gripper_change"))),
    ]
    return f'<h4 class="sub">{esc(title)}</h4>{stat_chips(pairs)}'


# --------------------------------------------------------------------------- #
# section builders
# --------------------------------------------------------------------------- #
def section_overview(traj, img, ood, epi):
    # aggregate
    rows = []
    total_eps = total_frames = 0
    seen = {}
    for t in traj:
        s = t["summary"]
        if t["is_bimanual"]:
            frames = (s["left"].get("frame_count") or 0)
            robot = s.get("robot_type", "bimanual")
            dim = len(s.get("left_joint_names", [])) + len(s.get("right_joint_names", []))
        else:
            frames = s.get("frame_count") or 0
            robot = "single-arm" if s.get("position_dim", 0) >= 3 else "2D"
            dim = s.get("position_dim")
        eps = s.get("episode_count") or 0
        seen[t["name"]] = {"episodes": eps, "frames": frames, "robot": robot, "dim": dim}
        total_eps += eps
        total_frames += frames

    for name, v in sorted(seen.items()):
        rows.append(
            f"<tr><td>{esc(name)}</td><td>{esc(v['robot'])}</td>"
            f"<td>{fmt(v['dim'])}</td><td class='num'>{fmt(v['episodes'])}</td>"
            f"<td class='num'>{fmt(v['frames'])}</td></tr>"
        )
    table = (
        "<table class='tbl'><thead><tr><th>Dataset</th><th>Robot</th>"
        "<th>State dim</th><th>Episodes</th><th>Frames</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    n_img = len(img["per_dataset"]) if img else 0
    kpis = [
        ("Datasets", len(seen), "trajectory-analyzed"),
        ("Total episodes", f"{total_eps:,}", "across datasets"),
        ("Total frames", f"{total_frames:,}", "state samples"),
        ("Image sets", n_img, "DINOv3 embedded"),
        ("OOD runs", len(ood), "train/test splits"),
        ("Episode inspections", len(epi), "ranked reports"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-val">{esc(v)}</div>'
        f'<div class="kpi-label">{esc(label)}</div>'
        f'<div class="kpi-sub">{esc(sub)}</div></div>'
        for label, v, sub in kpis
    )

    model = (img or {}).get("summary", {}).get("model_name", "DINOv3")
    return f"""
    <section id="overview" class="panel active">
      <div class="panel-head">
        <h2>Overview</h2>
        <p class="lead">LeRobot-format dataset trajectory &amp; distribution analysis. Robot motion,
        first-frame visual embeddings, and out-of-distribution signals across {len(seen)} datasets.</p>
      </div>
      <div class="kpis">{kpi_html}</div>
      <div class="card">
        <h3>Datasets</h3>
        {table}
      </div>
      <div class="note">Visual embeddings via <code>{esc(model)}</code>. OOD reports surface multiple
      signals rather than a single label — read the pattern across bars.</div>
    </section>
    """


def section_trajectory(traj):
    cards = []
    for t in traj:
        s = t["summary"]
        if t["is_bimanual"]:
            head = stat_chips([
                ("episodes", fmt(s.get("episode_count"))),
                ("robot", s.get("robot_type", "aloha")),
                ("FK", "yes" if s.get("has_fk") else "no"),
                ("state key", s.get("state_key", "–")),
            ])
            body = img_grid(t.get("combined", []))
            body += '<div class="armcols">'
            body += f'<div class="armcol">{speed_block("Left arm", s["left"])}{img_grid(t.get("left", []))}</div>'
            body += f'<div class="armcol">{speed_block("Right arm", s["right"])}{img_grid(t.get("right", []))}</div>'
            body += "</div>"
            badge = "bimanual"
        else:
            head = stat_chips([
                ("episodes", fmt(s.get("episode_count"))),
                ("frames", fmt(s.get("frame_count"))),
                ("dim", fmt(s.get("position_dim"))),
                ("coords", ",".join(s.get("coord_labels", []))),
                ("speed mean", fmt(s.get("speed_mean"))),
                ("speed p95", fmt(s.get("speed_p95"))),
                ("pos range", f'{vec(s.get("position_min"))} → {vec(s.get("position_max"))}'),
                ("grip-change eps", fmt(s.get("episodes_with_gripper_change"))),
            ])
            body = img_grid(t.get("images", []))
            notes = s.get("config_notes") or []
            if notes:
                body += '<div class="cfg-notes">' + "".join(
                    f"<span>• {esc(n)}</span>" for n in notes) + "</div>"
            badge = f'{s.get("position_dim", "?")}D'

        cards.append(
            f'<div class="card" data-name="{esc(t["name"])}">'
            f'<div class="card-title"><h3>{esc(t["name"])}</h3>'
            f'<span class="badge">{esc(badge)}</span></div>'
            f"{head}{body}</div>"
        )
    return f"""
    <section id="trajectory" class="panel">
      <div class="panel-head"><h2>Trajectory Distribution</h2>
        <p class="lead">End-effector position &amp; speed distributions. Color = episode progress
        (blue→red), cyan ▲ = first frame, red × = gripper change.</p></div>
      {''.join(cards)}
    </section>
    """


def section_image(img):
    if not img:
        return '<section id="image" class="panel"><div class="panel-head"><h2>Image Distribution</h2></div><div class="empty">No image distribution outputs found.</div></section>'
    s = img["summary"]
    g = s.get("global_embedding", {})
    head = stat_chips([
        ("model", s.get("model_name", "–")),
        ("samples", fmt(s.get("sample_count"))),
        ("embed dim", fmt(s.get("embedding_dim"))),
        ("reducer", g.get("reducer", "–")),
        ("clusters", fmt(g.get("cluster_count"))),
    ])
    global_card = (
        '<div class="card"><div class="card-title"><h3>Global embedding space</h3>'
        '<span class="badge">all datasets</span></div>'
        f"{head}{img_grid(img['global_images'])}</div>"
    )

    per_cards = []
    for d in img["per_dataset"]:
        meta = d["meta"]
        emb = d["embedding"]
        head = stat_chips([
            ("samples", fmt(meta.get("sample_count"))),
            ("clusters", fmt(emb.get("cluster_count"))),
            ("image keys", str(len(meta.get("image_keys", [])))),
            ("cam pose", ",".join(meta.get("camera_pose_sources", []) ) or "–"),
        ])
        keys = meta.get("image_keys", [])
        keys_html = '<div class="cfg-notes">' + "".join(
            f"<span>• {esc(k)}</span>" for k in keys) + "</div>" if keys else ""
        per_cards.append(
            f'<div class="card" data-name="{esc(d["name"])}">'
            f'<div class="card-title"><h3>{esc(d["name"])}</h3></div>'
            f"{head}{keys_html}{img_grid(d['images'])}</div>"
        )
    return f"""
    <section id="image" class="panel">
      <div class="panel-head"><h2>Image Distribution</h2>
        <p class="lead">First-frame DINOv3 embeddings projected with UMAP, colored by KMeans cluster,
        dataset, and camera key. Contact sheets show representative frames per cluster.</p></div>
      {global_card}
      <h3 class="group-title">Per-dataset</h3>
      {''.join(per_cards)}
    </section>
    """


def section_ood(ood):
    if not ood:
        return '<section id="ood" class="panel"><div class="panel-head"><h2>OOD Detection</h2></div><div class="empty">No OOD detection outputs found.</div></section>'
    cards = []
    for r in ood:
        s = r["summary"]
        split = s.get("split", {})
        head = stat_chips([
            ("mode", s.get("mode", "–")),
            ("train eps", fmt(split.get("train_episode_count"))),
            ("test eps", fmt(split.get("test_episode_count"))),
            ("train ratio", fmt(split.get("train_ratio"))),
            ("seed", fmt(split.get("seed"))),
        ])
        metrics = s.get("shift_metrics", {})
        bars = signal_bars(metrics) if metrics else ""
        cards.append(
            f'<div class="card" data-name="{esc(r["name"])}">'
            f'<div class="card-title"><h3>{esc(r["name"])}</h3>'
            f'<span class="badge">{esc(s.get("mode","").replace("_"," "))}</span></div>'
            f"{head}"
            f'<h4 class="sub">OOD signal strength</h4>{bars}'
            f"{img_grid(r['images'])}</div>"
        )
    legend = (
        '<div class="note"><b>Signal scale:</b> '
        '<span class="lg ok">0–0.5 similar</span>'
        '<span class="lg mod">0.5–1.5 moderate</span>'
        '<span class="lg strong">1.5+ strong OOD</span></div>'
    )
    return f"""
    <section id="ood" class="panel">
      <div class="panel-head"><h2>OOD Detection</h2>
        <p class="lead">Train vs test distribution comparison across trajectory, speed, initial pose,
        camera pose and first-frame embeddings. Multiple signals, no single yes/no label.</p></div>
      {legend}
      {''.join(cards)}
    </section>
    """


def section_episode(epi):
    if not epi:
        return '<section id="episode" class="panel"><div class="panel-head"><h2>OOD Episode Inspector</h2></div><div class="empty">No episode inspector outputs found.</div></section>'
    cards = []
    for r in epi:
        s = r["summary"]
        head = stat_chips([
            ("mode", s.get("mode", "–")),
            ("train eps", fmt(s.get("train_episode_count"))),
            ("test eps", fmt(s.get("test_episode_count"))),
            ("train score μ", fmt(s.get("train_score_mean"))),
            ("train p95", fmt(s.get("train_score_p95"))),
            ("test score μ", fmt(s.get("test_score_mean"))),
            ("test p95", fmt(s.get("test_score_p95"))),
        ])
        # top episodes table
        tops = s.get("top_episodes", [])[:12]
        sig_keys = list((tops[0]["signal_scores"].keys())) if tops else []
        ths = "".join(f"<th>{esc(k[:4])}</th>" for k in sig_keys)
        trs = []
        for e in tops:
            sig = e.get("signal_scores", {})
            sig_tds = "".join(
                f"<td class='num {_score_cls(sig.get(k))}'>{fmt(sig.get(k),2)}</td>"
                for k in sig_keys
            )
            trs.append(
                f"<tr><td class='num'>{esc(e.get('episode_id'))}</td>"
                f"<td class='num strong-score'>{fmt(e.get('total_score'),3)}</td>"
                f"{sig_tds}"
                f"<td class='reason'>{esc(e.get('reason',''))}</td></tr>"
            )
        table = (
            "<div class='tbl-wrap'><table class='tbl'><thead><tr><th>Ep</th><th>Score</th>"
            f"{ths}<th>Reason</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
        )
        cards.append(
            f'<div class="card" data-name="{esc(r["name"])}">'
            f'<div class="card-title"><h3>{esc(r["name"])}</h3>'
            f'<span class="badge">{esc(s.get("mode","").replace("_"," "))}</span></div>'
            f"{head}"
            f'<h4 class="sub">Top OOD episodes</h4>{table}'
            f"{img_grid(r['images'])}</div>"
        )
    return f"""
    <section id="episode" class="panel">
      <div class="panel-head"><h2>OOD Episode Inspector</h2>
        <p class="lead">Individual test episodes ranked by normalized OOD score, with the signal that
        drove each. Episodes beyond the train p95 are stronger OOD candidates.</p></div>
      {''.join(cards)}
    </section>
    """


def _score_cls(v):
    if v is None:
        return ""
    v = float(v)
    return "ok" if v < 0.5 else ("mod" if v < 1.5 else "strong")


# --------------------------------------------------------------------------- #
# page assembly
# --------------------------------------------------------------------------- #
NAV = [
    ("overview", "Overview"),
    ("trajectory", "Trajectory"),
    ("image", "Image Distribution"),
    ("ood", "OOD Detection"),
    ("episode", "Episode Inspector"),
]


def build():
    traj = collect_trajectory()
    img = collect_image_distribution()
    ood = collect_ood_detection()
    epi = collect_episode_inspector()

    nav_html = "".join(
        f'<button class="nav-item{" active" if i == 0 else ""}" data-target="{tid}">{esc(label)}</button>'
        for i, (tid, label) in enumerate(NAV)
    )

    sections = "".join([
        section_overview(traj, img, ood, epi),
        section_trajectory(traj),
        section_image(img),
        section_ood(ood),
        section_episode(epi),
    ])

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = TEMPLATE.format(nav=nav_html, sections=sections, css=CSS, js=JS, stamp=stamp)
    out = os.path.join(HERE, "index.html")
    with open(out, "w") as f:
        f.write(page)
    print(f"✓ dashboard written: {out}")
    print(f"  trajectory={len(traj)}  image_sets={len(img['per_dataset']) if img else 0}"
          f"  ood_runs={len(ood)}  episode_runs={len(epi)}")
    print("  open it directly, or serve:  python3 -m http.server -d dashboard 8000")


# --------------------------------------------------------------------------- #
# static assets (CSS / JS / shell)
# --------------------------------------------------------------------------- #
CSS = """
:root, [data-theme="dark"]{
  --bg:#0e1117; --bg2:#161b22; --card:#1b222c; --line:#2a323d;
  --txt:#e6edf3; --mut:#8b97a7; --acc:#4f9cf9; --acc2:#7ee787;
  --ok:#3fb950; --mod:#d29922; --strong:#f85149; --radius:14px;
  --shadow:none; --img-bg:#fff;
}
[data-theme="light"]{
  --bg:#f4f6f9; --bg2:#ffffff; --card:#ffffff; --line:#e2e7ee;
  --txt:#1f2733; --mut:#62707f; --acc:#1f6feb; --acc2:#1a7f37;
  --ok:#1a7f37; --mod:#9a6700; --strong:#cf222e;
  --shadow:0 1px 3px rgba(27,40,60,.06),0 1px 2px rgba(27,40,60,.04); --img-bg:#fbfcfd;
}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--txt);-webkit-font-smoothing:antialiased;
  transition:background .25s ease,color .25s ease}
.side,.card,.kpi,.chip,.bar-track,.note,.tbl th,.tbl td{transition:background .25s ease,border-color .25s ease,color .25s ease}
a{color:var(--acc)} code{background:var(--bg2);padding:1px 6px;border-radius:6px;font-size:.86em;color:var(--acc2)}
.layout{display:flex;min-height:100vh}
/* sidebar */
.side{width:240px;flex:0 0 240px;background:var(--bg2);border-right:1px solid var(--line);
  padding:26px 18px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column;gap:6px}
.brand{display:flex;align-items:center;gap:10px;margin-bottom:22px}
.brand .logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,var(--acc),#a371f7);
  display:grid;place-items:center;font-weight:800;color:#fff;font-size:17px}
.brand h1{font-size:16px;margin:0;letter-spacing:.2px}
.brand small{display:block;color:var(--mut);font-weight:400;font-size:11.5px}
.nav-item{all:unset;cursor:pointer;padding:9px 13px;border-radius:9px;color:var(--mut);
  font-size:14px;font-weight:500;transition:.15s;display:block}
.nav-item:hover{background:var(--card);color:var(--txt)}
.nav-item.active{background:rgba(79,156,249,.15);color:var(--acc)}
.theme-toggle{all:unset;cursor:pointer;margin-top:auto;display:flex;align-items:center;gap:10px;
  padding:9px 13px;border-radius:9px;color:var(--mut);font-size:13.5px;font-weight:500;
  border:1px solid var(--line);transition:.15s}
.theme-toggle:hover{background:var(--card);color:var(--txt)}
.theme-toggle .ic{font-size:15px;line-height:1}
.side .foot{color:var(--mut);font-size:11px;line-height:1.5;margin-top:14px}
/* main */
.main{flex:1;min-width:0;padding:34px 40px 80px;max-width:1280px}
.panel{display:none;animation:fade .25s ease} .panel.active{display:block}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.panel-head{margin-bottom:22px}
.panel-head h2{font-size:26px;margin:0 0 6px}
.lead{color:var(--mut);margin:0;max-width:760px}
.group-title{margin:30px 0 4px;font-size:15px;color:var(--mut);text-transform:uppercase;letter-spacing:1px}
/* kpis */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;margin-bottom:24px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:18px;box-shadow:var(--shadow)}
.kpi-val{font-size:28px;font-weight:700;letter-spacing:-.5px}
.kpi-label{font-size:13px;margin-top:2px}
.kpi-sub{font-size:11.5px;color:var(--mut);margin-top:2px}
/* cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  padding:22px;margin-bottom:20px;box-shadow:var(--shadow)}
.card h3{margin:0;font-size:18px}
.card-title{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.badge{font-size:11px;font-weight:600;color:var(--acc);background:rgba(79,156,249,.13);
  padding:3px 10px;border-radius:20px}
.sub{margin:20px 0 10px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.8px}
/* chips */
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.chip{background:var(--bg2);border:1px solid var(--line);border-radius:9px;padding:6px 11px;font-size:12.5px}
.chip .k{color:var(--mut)} .chip .v{margin-left:7px;font-weight:600}
.cfg-notes{margin-top:10px;display:flex;flex-direction:column;gap:3px;color:var(--mut);font-size:12.5px}
/* image grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px;margin-top:14px}
.shot{margin:0;cursor:zoom-in;background:var(--bg2);border:1px solid var(--line);border-radius:10px;
  overflow:hidden;transition:.15s}
.shot:hover{border-color:var(--acc);transform:translateY(-2px)}
.shot img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:var(--img-bg)}
.shot figcaption{padding:7px 10px;font-size:11.5px;color:var(--mut);border-top:1px solid var(--line)}
.armcols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:8px}
@media(max-width:900px){.armcols{grid-template-columns:1fr}}
.armcol h4{margin:0 0 8px}
/* tables */
.tbl-wrap{overflow-x:auto}
.tbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
.tbl th{text-align:left;color:var(--mut);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line);
  white-space:nowrap}
.tbl td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.tbl tr:hover td{background:rgba(255,255,255,.02)}
.tbl .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl .reason{color:var(--mut);font-size:12px;min-width:280px}
.strong-score{font-weight:700;color:var(--acc)}
td.ok{color:var(--ok)} td.mod{color:var(--mod)} td.strong{color:var(--strong)}
/* bars */
.bars{display:flex;flex-direction:column;gap:9px;margin:6px 0 4px}
.bar-row{display:grid;grid-template-columns:230px 1fr 62px;align-items:center;gap:12px;font-size:12.5px}
.bar-label{color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-track{background:var(--bg2);border-radius:20px;height:12px;overflow:hidden;border:1px solid var(--line)}
.bar-fill{height:100%;border-radius:20px}
.bar-fill.ok{background:var(--ok)} .bar-fill.mod{background:var(--mod)} .bar-fill.strong{background:var(--strong)}
.bar-val{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
@media(max-width:700px){.bar-row{grid-template-columns:120px 1fr 52px}}
/* note */
.note{background:var(--bg2);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:10px;padding:12px 16px;font-size:13px;color:var(--mut);margin:8px 0 20px;
  display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.lg{padding:2px 9px;border-radius:20px;font-size:11.5px;font-weight:600;color:#0e1117}
.lg.ok{background:var(--ok)} .lg.mod{background:var(--mod)} .lg.strong{background:var(--strong)}
.empty{color:var(--mut);padding:40px;text-align:center;background:var(--card);border-radius:var(--radius)}
/* lightbox */
.lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;place-items:center;z-index:99;cursor:zoom-out}
.lb.show{display:grid} .lb img{max-width:94vw;max-height:90vh;border-radius:8px;background:#fff}
.lb .cap{position:absolute;bottom:22px;color:#fff;font-size:13px;background:rgba(0,0,0,.5);padding:6px 14px;border-radius:20px}
"""

JS = """
// theme toggle (persisted; defaults to system preference, else dark)
const root=document.documentElement, tBtn=document.getElementById('themeToggle');
function applyTheme(t){
  root.setAttribute('data-theme',t);
  if(tBtn){tBtn.querySelector('.ic').textContent = t==='dark'?'\\u2600':'\\u263D';
    tBtn.querySelector('.lbl').textContent = t==='dark'?'Light mode':'Dark mode';}
}
const saved=localStorage.getItem('rt-theme');
const sysLight=window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches;
applyTheme(saved || (sysLight?'light':'dark'));
tBtn&&tBtn.addEventListener('click',()=>{
  const next=root.getAttribute('data-theme')==='dark'?'light':'dark';
  applyTheme(next); localStorage.setItem('rt-theme',next);
});

const items=document.querySelectorAll('.nav-item');
const panels=document.querySelectorAll('.panel');
items.forEach(b=>b.addEventListener('click',()=>{
  const t=b.dataset.target;
  items.forEach(x=>x.classList.toggle('active',x===b));
  panels.forEach(p=>p.classList.toggle('active',p.id===t));
  window.scrollTo({top:0,behavior:'smooth'});
  history.replaceState(null,'','#'+t);
}));
// deep-link support
const h=location.hash.slice(1);
if(h){const b=[...items].find(x=>x.dataset.target===h); if(b) b.click();}
// lightbox
const lb=document.createElement('div');lb.className='lb';
lb.innerHTML='<img><div class="cap"></div>';document.body.appendChild(lb);
lb.addEventListener('click',()=>lb.classList.remove('show'));
function zoom(fig){const img=fig.querySelector('img');
  lb.querySelector('img').src=img.src;
  lb.querySelector('.cap').textContent=fig.querySelector('figcaption')?.textContent||'';
  lb.classList.add('show');}
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('show')});
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboTracker · Dataset Analysis Dashboard</title>
<script>
  (function(){{var s=localStorage.getItem('rt-theme');
    var sys=window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches;
    document.documentElement.setAttribute('data-theme', s||(sys?'light':'dark'));}})();
</script>
<style>{css}</style>
</head><body>
<div class="layout">
  <aside class="side">
    <div class="brand">
      <div class="logo">R</div>
      <div><h1>RoboTracker</h1><small>Dataset Analysis</small></div>
    </div>
    <nav>{nav}</nav>
    <button class="theme-toggle" id="themeToggle" type="button" aria-label="Toggle color theme">
      <span class="ic">☀</span><span class="lbl">Light mode</span>
    </button>
    <div class="foot">LeRobot trajectory analyzer<br>Generated {stamp}<br>
      <code>build_dashboard.py</code></div>
  </aside>
  <main class="main">
    {sections}
  </main>
</div>
<script>{js}</script>
</body></html>
"""


if __name__ == "__main__":
    build()

"""PDF work orders with reportlab (standard fonts, so rupees are written as 'Rs')."""

import io
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .content import fmt_inr

GREEN = colors.HexColor("#0d7a5f")
GREY = colors.HexColor("#64748b")

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, textColor=colors.black)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=GREEN, spaceBefore=8)
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, textColor=GREY, leading=9.5)


def _p(text: Any, style=BODY) -> Paragraph:
    s = "" if text is None else str(text)
    return Paragraph(s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("₹", "Rs "), style)


def _table(rows: List[List[Any]], widths: List[float]) -> Table:
    t = Table([[_p(c, SMALL if r == 0 else BODY) for c in row] for r, row in enumerate(rows)], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, GREEN),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _intervention_rows(lines: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = [["Intervention", "Side", "Quantity", "Cost low / central / high", "Cooling C (literature)", "Phase", "Permits and conditions"]]
    for ln in lines:
        cool = "-" if ln["cooling_c_central"] is None else f"{ln['cooling_c_low']:.1f} / {ln['cooling_c_central']:.1f} / {ln['cooling_c_high']:.1f}"
        rows.append([
            ln["intervention"],
            (ln["side"] or "site").replace("_", " "),
            f"{ln['quantity']:g} {ln['unit']} ({ln['quantity_source']})",
            f"{fmt_inr(ln['cost_inr_low'])} / {fmt_inr(ln['cost_inr_central'])} / {fmt_inr(ln['cost_inr_high'])}",
            cool,
            ln["phase"],
            "; ".join(ln["permits"] + [f"Verify: {c}" for c in ln["conditions_to_verify"]]),
        ])
    return rows


def _rejected_rows(rejected: List[Dict[str, Any]]) -> List[List[Any]]:
    rows = [["Option", "Side", "Rule", "Reason", "Measured"]]
    for r in rejected:
        measured = ", ".join(f"{k}={v}" for k, v in (r.get("measured") or {}).items())
        rows.append([r["type"].replace("_", " "), (r.get("target_zone") or "site").replace("_", " "), r.get("rule"), r["reason"], measured])
    return rows


def _model_cards(order: Dict[str, Any]) -> List[Any]:
    cards = order["model_cards"]
    site = cards["site_model"]["spatial_cv"] or {}
    legacy = cards["legacy_land_cover_model"]["spatial_cv"] or {}
    fmt = lambda m: ", ".join(f"{k} {v:.3f}" for k, v in m.items()) if m else "not available"  # noqa: E731
    return [
        _p("Model cards", H2),
        _p(f"Site heat model (hotspots; spatial CV): {fmt(site)}"),
        _p(f"Legacy land-cover model (low confidence, not used for intervention cooling; spatial CV): {fmt(legacy)}"),
    ]


def render_pdf(order: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="ResiliCity work order")
    width = A4[0] - 28 * mm
    w_int = [w * width for w in (0.17, 0.08, 0.13, 0.2, 0.12, 0.1, 0.2)]
    w_rej = [w * width for w in (0.14, 0.09, 0.17, 0.4, 0.2)]
    story: List[Any] = []

    if order["kind"] == "project_work_order":
        story += [_p(f"Work order: {order['project']['name']}", H1),
                  _p(f"Generated {order['generated']}. Data mode: {order['data_mode']}.", SMALL)]
        hs = order.get("hotspot")
        if hs:
            story.append(_p(f"Hotspot rank {hs['rank']} (cell {hs['cell_id']}): predicted surface temperature "
                            f"{hs['lst_pred_c']} C, 90% interval {hs['lst_interval_90_c']} (model); population "
                            f"{hs['population']} ({hs['sources']['population']})."))
        st = order["street"]
        story.append(_p(f"Street: carriageway {st['carriageway_m']} m ({st['carriageway_source']}), footpaths "
                        f"{st['left_footpath_m']} / {st['right_footpath_m']} m, +/-{st['uncertainty_pct']}% (measured from photo)."))
        story += [_p(f"Interventions per {order['street_length_m']:g} m of street", H2),
                  _table(_intervention_rows(order["interventions"]), w_int)]
        t = order["totals"]
        story.append(_p(f"Total {fmt_inr(t['cost_inr_low'])} to {fmt_inr(t['cost_inr_high'])} (assumed rates); "
                        f"indicative local cooling {t['cooling_c_central']} C (literature)."))
        if order["rejected"]:
            story += [_p("Considered and rejected", H2), _table(_rejected_rows(order["rejected"]), w_rej)]
    else:
        t = order["totals"]
        story += [_p("Portfolio work order", H1),
                  _p(f"Generated {order['generated']}. Objective {order['objective']}, budget Rs {order['budget_lakh']} L. "
                     f"Data mode: {', '.join(order['data_mode'])}.", SMALL),
                  _p(f"Selected {t['n_items']} items: {fmt_inr(t['cost_inr']['central'])} central cost "
                     f"({fmt_inr(t['cost_inr']['low'])} to {fmt_inr(t['cost_inr']['high'])}); "
                     f"{t['person_c']['central']:.0f} person-C (band {t['person_c']['low']:.0f} to {t['person_c']['high']:.0f}); "
                     f"{t['c_per_lakh']:.3f} C per Rs lakh.")]
        for scen, r in (order.get("robustness") or {}).items():
            same = r["same_selection"]
            story.append(_p(f"{scen.title()} case ({r['assumption']}): cost {fmt_inr(same['cost_inr'])}, "
                            f"{same['person_c']:.0f} person-C, within budget: {'yes' if same['within_budget'] else 'no'}."))
        if order.get("phase_schedule"):
            story.append(_p("Phase schedule: " + "; ".join(
                f"{s['label']}: {len(s['items'])} items, {fmt_inr(s['cost_inr_central'])}" for s in order["phase_schedule"])))
        for proj in order["projects"]:
            story += [_p(f"{proj['project']['name']} (hotspot rank {proj['hotspot_rank']})", H2),
                      _table(_intervention_rows(proj["interventions"]), w_int)]
            if proj["rejected"]:
                story += [Spacer(1, 3), _p("Rejected options", SMALL), _table(_rejected_rows(proj["rejected"]), w_rej)]

    story += _model_cards(order)
    story += [_p("Disclaimer", H2), _p(order["disclaimer"])]
    doc.build(story)
    return buf.getvalue()

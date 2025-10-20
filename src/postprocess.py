def normalize_row(row: dict) -> dict:
    resp = (row.get("responsable") or "").strip()
    resp = " ".join(resp.split())
    row["responsable"] = resp

    row["fecha_iso"] = (row.get("fecha_iso") or "").strip()
    return row

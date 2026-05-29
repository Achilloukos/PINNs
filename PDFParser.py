import argparse
import importlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _require_module(module_name: str):
	try:
		return importlib.import_module(module_name)
	except ModuleNotFoundError as exc:
		raise SystemExit(
			f"Missing dependency '{module_name}'. Install dependencies first, e.g.: pip install pymupdf pillow"
		) from exc


def _require_pymupdf_module():
	# Newer PyMuPDF versions expose the module as 'pymupdf'; older code often uses 'fitz'.
	for module_name in ("pymupdf", "fitz"):
		try:
			return importlib.import_module(module_name)
		except ModuleNotFoundError:
			continue
	raise SystemExit(
		"Missing PyMuPDF module. Install it with: pip install pymupdf"
	)


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def extract_field_from_lines(lines: Sequence[str], labels: Sequence[str]) -> Optional[str]:
	for line in lines:
		for label in labels:
			pattern = rf"(?i){re.escape(label)}\s*[:\-]?\s*(.*)$"
			match = re.search(pattern, line)
			if match:
				value = normalize_text(match.group(1))
				if value:
					return value
	return None


def extract_materialguete_from_lines(lines: Sequence[str]) -> Optional[str]:
	pattern = re.compile(r"(?i)Materialg(?:ü|ue)te(?:\s*\(n\))?\s*[:\-]\s*(.+)$")
	for line in lines:
		match = pattern.search(line)
		if match:
			value = normalize_text(match.group(1))
			if value:
				return value
	return None


def extract_field(full_text: str, labels: Sequence[str]) -> Optional[str]:
	# Fallback extractor kept for labels that are reliably on the same text line in full text.
	for label in labels:
		pattern = rf"(?im){re.escape(label)}\s*[:\-]?\s*(.+)$"
		match = re.search(pattern, full_text)
		if match:
			value = normalize_text(match.group(1))
			if value:
				return value
	return None


def extract_value_right_of_label(
	doc: Any,
	label_patterns: Sequence[str],
	y_tol: float = 9.0,
) -> Optional[str]:
	compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in label_patterns]

	for page in doc:
		words = page.get_text("words")
		for label_word in words:
			label_text = str(label_word[4])
			if not any(pattern.search(label_text) for pattern in compiled):
				continue

			label_x1 = float(label_word[2])
			label_y = _word_center_y(label_word)

			right_words = [
				w
				for w in words
				if float(w[0]) > label_x1 + 2.0 and abs(_word_center_y(w) - label_y) <= y_tol
			]
			if not right_words:
				continue

			right_words.sort(key=lambda w: float(w[0]))
			value = normalize_text(" ".join(str(w[4]) for w in right_words))
			if value:
				return value

	return None


def extract_value_from_same_block_next_line(doc: Any, label_pattern: str) -> Optional[str]:
	compiled = re.compile(label_pattern, flags=re.IGNORECASE)

	for page in doc:
		for block in page.get_text("dict").get("blocks", []):
			if block.get("type") != 0:
				continue

			line_texts: List[str] = []
			for line in block.get("lines", []):
				text = normalize_text("".join(span.get("text", "") for span in line.get("spans", [])))
				line_texts.append(text)

			for idx, text in enumerate(line_texts):
				if not compiled.search(text):
					continue

				for j in range(idx + 1, len(line_texts)):
					candidate = line_texts[j]
					if candidate and not compiled.search(candidate):
						return candidate

				for j in range(idx - 1, -1, -1):
					candidate = line_texts[j]
					if candidate and not compiled.search(candidate):
						return candidate

	return None


def get_document_lines(doc: Any) -> List[str]:
	lines_out: List[str] = []
	for page in doc:
		blocks = page.get_text("dict").get("blocks", [])
		for block in blocks:
			if block.get("type") != 0:
				continue
			for line in block.get("lines", []):
				spans = line.get("spans", [])
				text = normalize_text("".join(span.get("text", "") for span in spans))
				if text:
					lines_out.append(text)
	return lines_out


def find_heading_y(page: Any, heading_text: str) -> Optional[float]:
	words = page.get_text("words")
	if not words:
		return None

	target = normalize_text(heading_text).lower()
	full_lines: Dict[Tuple[int, int], List[Tuple[float, float, str]]] = {}
	for x0, y0, x1, y1, word, block_no, line_no, _ in words:
		full_lines.setdefault((block_no, line_no), []).append((x0, y0, word))

	for parts in full_lines.values():
		parts.sort(key=lambda x: x[0])
		line_text = normalize_text(" ".join(p[2] for p in parts)).lower()
		if target in line_text:
			return min(p[1] for p in parts)
	return None


def _normalize_token(token: str) -> str:
	t = normalize_text(token).lower()
	t = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
	return t


def find_heading_y_by_tokens(page: Any, required_tokens: Sequence[str]) -> Optional[float]:
	words = page.get_text("words")
	if not words:
		return None

	required = {_normalize_token(token) for token in required_tokens}
	rows: Dict[float, List[Tuple[Any, ...]]] = {}
	for w in words:
		y_center = round(_word_center_y(w) / 3.0) * 3.0
		rows.setdefault(y_center, []).append(w)

	for y_key in sorted(rows.keys()):
		row_words = rows[y_key]
		token_set = {_normalize_token(str(w[4])) for w in row_words}
		if required.issubset(token_set):
			return min(float(w[1]) for w in row_words)
	return None


def classify_status_icon(image_bytes: bytes) -> Optional[str]:
	pil_image_module = _require_module("PIL.Image")
	try:
		img = pil_image_module.open(io.BytesIO(image_bytes)).convert("RGB")
	except Exception:
		return None

	pixels = list(img.getdata())
	non_white = [p for p in pixels if not (p[0] > 245 and p[1] > 245 and p[2] > 245)]
	if not non_white:
		return None

	red_count = 0
	green_count = 0
	yellow_count = 0
	for r, g, b in non_white:
		if g > r * 1.15 and g > b * 1.1:
			green_count += 1
		if r > g * 1.15 and r > b * 1.1:
			red_count += 1
		if r > 130 and g > 130 and b < 145:
			yellow_count += 1

	total = len(non_white)
	votes = {
		"red_cross": red_count,
		"green_check": green_count,
		"yellow_warning": yellow_count,
	}
	best_label = max(votes, key=votes.get)
	best_count = votes[best_label]
	if best_count < max(18, int(0.015 * total)):
		return None

	return best_label


def status_to_user_value(status: Optional[str]) -> str:
	if status == "green_check":
		return "pass"
	if status == "red_cross":
		return "error"
	if status == "yellow_warning":
		return "warning"
	return "-"


def get_image_bytes_from_xref(doc: Any, xref: int) -> Optional[bytes]:
	try:
		info = doc.extract_image(xref)
		return info.get("image")
	except Exception:
		return None


def get_image_bytes_from_block(doc: Any, block: Dict[str, Any]) -> Tuple[Optional[bytes], str]:
	# Some PDFs embed images inline in text blocks without exposing an xref.
	xref = block.get("xref")
	if xref:
		img_bytes = get_image_bytes_from_xref(doc, xref)
		if img_bytes:
			return img_bytes, "png"

	img_bytes = block.get("image")
	ext = str(block.get("ext") or "png").lower()
	if isinstance(img_bytes, (bytes, bytearray)) and img_bytes:
		if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
			ext = "png"
		return bytes(img_bytes), ext

	return None, "png"


def _collect_images_from_page(
	doc: Any,
	page_index: int,
	y_heading: float,
	output_dir: Path,
	name_prefix: str,
	base_output_dir: Path,
) -> List[Dict[str, Any]]:
	output_dir.mkdir(parents=True, exist_ok=True)
	saved: List[Dict[str, Any]] = []
	page = doc[page_index]

	pymupdf = _require_pymupdf_module()
	blocks = page.get_text("dict").get("blocks", [])
	drawings = page.get_drawings()
	img_count = 0
	img_bottom_y = y_heading
	for block in blocks:
		if block.get("type") != 1:
			continue

		x0, y0, x1, y1 = block.get("bbox", [0, 0, 0, 0])
		width = x1 - x0
		height = y1 - y0
		area = width * height

		if y0 < y_heading:
			continue
		if area < 2_000:
			continue

		# Check if there are vector drawings overlaying this image
		has_overlays = any(
			d.get("rect") and
			d["rect"][0] >= x0 and d["rect"][1] >= y0 and
			d["rect"][2] <= x1 and d["rect"][3] <= y1
			for d in drawings
		)

		if has_overlays:
			# Render as pixmap to capture both raster image and vector overlays
			clip = pymupdf.Rect(x0, y0, x1, y1)
			pix = page.get_pixmap(clip=clip, dpi=150)
			img_count += 1
			file_name = f"{name_prefix}_p{page_index + 1}_{img_count}.png"
			out_path = output_dir / file_name
			pix.save(str(out_path))
		else:
			img_bytes, ext = get_image_bytes_from_block(doc, block)
			if not img_bytes:
				continue
			img_count += 1
			file_name = f"{name_prefix}_p{page_index + 1}_{img_count}.{ext}"
			out_path = output_dir / file_name
			out_path.write_bytes(img_bytes)

		saved.append(
			{
				"page": page_index + 1,
				"image_file": file_name,
				"image_rel_path": str(out_path.relative_to(base_output_dir)),
			}
		)
		if y1 > img_bottom_y:
			img_bottom_y = y1

	# Render colorbar/legend region below raster images (vector graphics)
	legend_draws = [
		d for d in drawings
		if d.get("rect") and d["rect"][1] >= img_bottom_y and d["rect"][3] <= page.rect.height - 20
	]
	if legend_draws:
		legend_y0 = min(d["rect"][1] for d in legend_draws)
		legend_y1 = max(d["rect"][3] for d in legend_draws)
		legend_x0 = min(d["rect"][0] for d in legend_draws)
		legend_x1 = max(d["rect"][2] for d in legend_draws)
		# Only save if the region has meaningful height (> 10pt)
		if legend_y1 - legend_y0 > 10:
			clip = pymupdf.Rect(legend_x0 - 2, legend_y0 - 2, legend_x1 + 2, legend_y1 + 2)
			pix = page.get_pixmap(clip=clip, dpi=150)
			img_count += 1
			file_name = f"{name_prefix}_p{page_index + 1}_{img_count}_legend.png"
			out_path = output_dir / file_name
			pix.save(str(out_path))
			saved.append(
				{
					"page": page_index + 1,
					"image_file": file_name,
					"image_rel_path": str(out_path.relative_to(base_output_dir)),
				}
			)

	return saved


def collect_section_images(
	doc: Any,
	output_dir: Path,
	required_heading_tokens: Sequence[str],
	name_prefix: str,
	base_output_dir: Path,
) -> List[Dict[str, Any]]:
	for page_index in range(len(doc)):
		page = doc[page_index]
		y_heading = find_heading_y_by_tokens(page, required_heading_tokens)
		if y_heading is not None:
			return _collect_images_from_page(doc, page_index, y_heading, output_dir, name_prefix, base_output_dir)
	return []


def find_diagram_sections(doc: Any) -> List[Dict[str, Any]]:
	"""Scan the document for Grenzformänderungsdiagramm/Formability and Kantenriss headings.

	Returns a list of dicts with keys: page_index, y_heading, number, process_name, section_type.
	"""
	sections: List[Dict[str, Any]] = []
	seen: set = set()

	for page_index in range(len(doc)):
		page = doc[page_index]
		words = page.get_text("words")
		if not words:
			continue

		rows: Dict[float, List[Any]] = {}
		for w in words:
			y_center = round(_word_center_y(w) / 3.0) * 3.0
			rows.setdefault(y_center, []).append(w)

		for y_key in sorted(rows.keys()):
			row_words = sorted(rows[y_key], key=lambda w: float(w[0]))
			tokens = [str(w[4]) for w in row_words]
			normalized_tokens = {_normalize_token(t) for t in tokens}

			is_fld = "formability" in normalized_tokens or any(
				"grenzformaenderungsdiagramm" in t for t in normalized_tokens
			)
			is_kantenriss = "kantenriss" in normalized_tokens

			if not is_fld and not is_kantenriss:
				continue

			# Extract the number prefix and process name from the heading line.
			# Expected pattern: "1. Ziehen Grenzformänderungsdiagramm | Formability"
			# or "2. Formstufe Kantenriss" etc.
			number = None
			process_parts: List[str] = []
			stop_tokens = {
				"grenzformaenderungsdiagramm", "formability", "kantenriss", "|",
			}

			for tok in tokens:
				if number is None and re.match(r"^\d+\.$", tok):
					number = tok.rstrip(".")
					continue
				if number is not None:
					if _normalize_token(tok) in stop_tokens:
						break
					process_parts.append(tok)

			if number is None:
				continue

			process_name = " ".join(process_parts) if process_parts else "unknown"
			section_type = "grenzformaenderungsdiagramm_formability" if is_fld else "kantenriss"

			key = (number, _normalize_token(process_name), section_type)
			if key in seen:
				continue
			seen.add(key)

			y_heading = min(float(w[1]) for w in row_words)

			sections.append({
				"page_index": page_index,
				"y_heading": y_heading,
				"number": number,
				"process_name": process_name,
				"section_type": section_type,
			})

	return sections


def _word_center_x(word: Tuple[Any, ...]) -> float:
	return (float(word[0]) + float(word[2])) / 2.0


def _word_center_y(word: Tuple[Any, ...]) -> float:
	return (float(word[1]) + float(word[3])) / 2.0


def derive_berechnungsergebnisse_columns(page: Any, y_heading: float) -> List[Dict[str, Any]]:
	words = page.get_text("words")
	header_words = [
		w for w in words if w[0] > 620 and (y_heading + 35) <= _word_center_y(w) <= (y_heading + 140)
	]

	number_words = sorted(
		[w for w in header_words if re.match(r"^[1-9]\.$", str(w[4]))],
		key=lambda w: w[0],
	)
	process_words = sorted(
		[w for w in header_words if str(w[4]) in {"Ziehen", "Nachformen"}],
		key=lambda w: w[0],
	)

	columns: List[Dict[str, Any]] = []
	pair_count = min(len(number_words), len(process_words))
	for idx in range(pair_count):
		num = str(number_words[idx][4]).rstrip(".")
		label = str(process_words[idx][4])
		x = (_word_center_x(number_words[idx]) + _word_center_x(process_words[idx])) / 2.0
		columns.append({"name": f"{num}. {label}", "x": x})

	for extra_label in ("Beschnitt", "Aufsprung", "Kosten","Prozesssicherheit"):
		candidates = [w for w in header_words if str(w[4]) == extra_label]
		if candidates:
			columns.append({"name": extra_label, "x": _word_center_x(candidates[0])})

	columns.sort(key=lambda c: c["x"])
	return columns


def _find_row_y(page: Any, row_label_pattern: str) -> Optional[float]:
	words = page.get_text("words")
	candidates = [w for w in words if re.search(row_label_pattern, str(w[4]), flags=re.IGNORECASE)]
	if not candidates:
		return None
	return min(_word_center_y(w) for w in candidates)


def _extract_icon_points_in_table(page: Any, doc: Any, y_heading: float) -> List[Dict[str, Any]]:
	icons: List[Dict[str, Any]] = []
	for block in page.get_text("dict").get("blocks", []):
		if block.get("type") != 1:
			continue

		x0, y0, x1, y1 = block.get("bbox", [0, 0, 0, 0])
		width = x1 - x0
		height = y1 - y0
		area = width * height

		if x0 < 620:
			continue
		if y0 < y_heading + 100 or y1 > y_heading + 240:
			continue
		if area < 80 or area > 12_000:
			continue

		img_bytes, _ = get_image_bytes_from_block(doc, block)
		if not img_bytes:
			continue

		status = classify_status_icon(img_bytes)

		icons.append(
			{
				"x": (x0 + x1) / 2.0,
				"y": (y0 + y1) / 2.0,
				"w": width,
				"h": height,
				"area": area,
				"status": status,
			}
		)
	return icons


def _column_bands(columns: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
	xs = [float(c["x"]) for c in columns]
	bands: List[Tuple[float, float]] = []
	for idx, x in enumerate(xs):
		left = (xs[idx - 1] + x) / 2.0 if idx > 0 else x - (xs[idx + 1] - x) / 2.0
		right = (x + xs[idx + 1]) / 2.0 if idx < len(xs) - 1 else x + (x - xs[idx - 1]) / 2.0
		bands.append((left, right))
	return bands


def _icons_in_cell(
	icons: Sequence[Dict[str, Any]],
	x_left: float,
	x_right: float,
	y_center: float,
	y_tol: float,
) -> List[Dict[str, Any]]:
	return [
		icon
		for icon in icons
		if x_left <= icon["x"] <= x_right and abs(icon["y"] - y_center) <= y_tol
	]


def _nearest_icon_status(
	icons: Sequence[Dict[str, Any]],
	x: float,
	y: float,
	x_tol: float = 18.0,
	y_tol: float = 16.0,
) -> Optional[str]:
	best = None
	best_dist = 1e18
	for icon in icons:
		dx = abs(icon["x"] - x)
		dy = abs(icon["y"] - y)
		if dx > x_tol or dy > y_tol:
			continue
		dist = dx * dx + dy * dy
		if dist < best_dist:
			best_dist = dist
			best = icon["status"]
	return best


def detect_berechnungsergebnisse_status(
	doc: Any,
	page: Any,
	y_heading: float,
) -> Dict[str, Any]:
	columns = derive_berechnungsergebnisse_columns(page, y_heading)
	if not columns:
		return {"columns": []}

	icons = _extract_icon_points_in_table(page, doc, y_heading)
	bands = _column_bands(columns)

	# Derive status row y from the large icons themselves (self-calibrating)
	large_icons = [icon for icon in icons if icon["h"] >= 25]
	status_row_y = sum(icon["y"] for icon in large_icons) / len(large_icons) if large_icons else None
	riss_y = _find_row_y(page, r"Riss")
	falten_y = _find_row_y(page, r"Falten|Wellen")
	kantenriss_y = _find_row_y(page, r"Kantenriss")

	rows: List[Dict[str, Any]] = []
	for idx, col in enumerate(columns):
		x_left, x_right = bands[idx]

		if status_row_y is not None:
			status_icons = _icons_in_cell(icons, x_left, x_right, status_row_y, y_tol=30)
			status_icons = [icon for icon in status_icons if icon["h"] >= 25]
			status_raw = status_icons[0]["status"] if status_icons else None
		else:
			status_raw = None

		riss_icons = _icons_in_cell(icons, x_left, x_right, riss_y, y_tol=11) if riss_y else []
		falten_icons = _icons_in_cell(icons, x_left, x_right, falten_y, y_tol=11) if falten_y else []
		kanten_icons = _icons_in_cell(icons, x_left, x_right, kantenriss_y, y_tol=11) if kantenriss_y else []

		# Only count as "check" if there is an icon with an actual classified status (not empty placeholders)
		has_riss = any(icon["status"] is not None for icon in riss_icons)
		has_falten = any(icon["status"] is not None for icon in falten_icons)
		has_kanten = any(icon["status"] is not None for icon in kanten_icons)

		rows.append(
			{
				"column": col["name"],
				"status": status_to_user_value(status_raw),
				"Riss/Rissgefahr": "check" if has_riss else "-",
				"Falten/Wellen": "check" if has_falten else "-",
				"Kantenriss": "check" if has_kanten else "-",
			}
		)

	return {"columns": rows}


def extract_pdf_data(pdf_path: Path, output_root: Path) -> Dict[str, Any]:
	fitz = _require_pymupdf_module()
	doc = fitz.open(pdf_path)
	all_text = "\n".join(page.get_text("text") for page in doc)
	all_lines = get_document_lines(doc)

	pdf_output_dir = output_root / pdf_path.stem

	berechnung_page = None
	berechnung_y = None
	berechnung_page_number = None
	for page_index in range(len(doc)):
		candidate_page = doc[page_index]
		y_heading = find_heading_y(candidate_page, "Berechnungsergebnisse")
		if y_heading is not None:
			berechnung_page = candidate_page
			berechnung_y = y_heading
			berechnung_page_number = page_index + 1
			break

	# Dynamically discover diagram sections (Grenzformänderungsdiagramm & Kantenriss)
	sections = find_diagram_sections(doc)
	images_data: Dict[str, Any] = {}
	for section in sections:
		proc_lower = _normalize_token(section["process_name"]).replace(" ", "_")
		num = section["number"]

		if section["section_type"] == "grenzformaenderungsdiagramm_formability":
			label = f"{num}. {section['process_name']} Grenzformänderungsdiagramm | Formability"
			folder_name = f"{num}_{proc_lower}_grenzformaenderungsdiagramm_formability"
			file_prefix = f"{num}_{proc_lower}_formability"
		else:
			label = f"{num}. {section['process_name']} Kantenriss"
			folder_name = f"{num}_{proc_lower}_kantenriss"
			file_prefix = f"{num}_{proc_lower}_kantenriss"

		images_data[label] = _collect_images_from_page(
			doc,
			section["page_index"],
			section["y_heading"],
			pdf_output_dir / "images" / folder_name,
			file_prefix,
			output_root,
		)

	data = {
		"source_pdf": str(pdf_path),
		"output_folder": str(pdf_output_dir),
		"fields": {
			"Name": extract_value_right_of_label(doc, [r"^Name:?$"]),
			"Bauteilname": extract_field_from_lines(all_lines, ["Bauteilname"])
			or extract_field(all_text, ["Bauteilname"]),
			"Prismanummer(n)/Importdatei(en)": extract_value_from_same_block_next_line(
				doc,
				r"^Prismanummer\(n\)/Importdatei\(en\):?$",
			)
			or extract_value_right_of_label(
				doc,
				[r"^Prismanummer\(n\)/Importdatei\(en\):?$"],
			)
			or None,
			"Materialgüte(n)": extract_value_right_of_label(
				doc,
				[r"^Materialg(?:ü|ue)te\(n\):?$", r"^Materialg(?:ü|ue)te:?$"],
			)
			or extract_materialguete_from_lines(all_lines),
			"Rechenblechdicke(n)": extract_value_right_of_label(
				doc,
				[r"^Rechenblechdicke\(n\):?$"],
			),
		},
		"images": images_data,
		"berechnungsergebnisse": {
			"columns": [],
		},
	}

	if berechnung_page is not None and berechnung_y is not None:
		data["berechnungsergebnisse"] = {
			**detect_berechnungsergebnisse_status(doc, berechnung_page, berechnung_y),
		}

	doc.close()
	return data


def list_pdfs(input_path: Path) -> List[Path]:
	if input_path.is_file() and input_path.suffix.lower() == ".pdf":
		return [input_path]
	if input_path.is_dir():
		return sorted(input_path.rglob("*.pdf"))
	return []


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Extract selected text fields, section images, and status icons from PDFs.",
	)
	parser.add_argument("input", nargs="?", default=None, help="PDF file or folder containing PDFs")
	parser.add_argument(
		"--pdf-name",
		help="PDF filename to process when input is a folder (e.g. report.pdf)",
	)
	parser.add_argument(
		"-o",
		"--output",
		default=None,
		help="Directory for extracted images and default JSON location",
	)
	parser.add_argument(
		"--json-file",
		default=None,
		help="Optional combined JSON path. If omitted, each PDF gets result.json in its image parent folder.",
	)
	args = parser.parse_args()

	input_path = Path(args.input if args.input else INPUT_PATH)
	output_dir = Path(args.output if args.output else OUTPUT_DIR)
	output_dir.mkdir(parents=True, exist_ok=True)

	if input_path.is_dir() and args.pdf_name:
		selected_pdf = input_path / args.pdf_name
		if not selected_pdf.exists() and not selected_pdf.suffix:
			selected_pdf = input_path / f"{args.pdf_name}.pdf"
		if not selected_pdf.exists() or selected_pdf.suffix.lower() != ".pdf":
			raise SystemExit(f"PDF not found in folder: {args.pdf_name}")
		pdfs = [selected_pdf]
	else:
		pdfs = list_pdfs(input_path)

	if not pdfs:
		raise SystemExit(f"No PDF files found in: {input_path}")

	all_results = []
	for pdf in pdfs:
		result = extract_pdf_data(pdf, output_dir)
		all_results.append(result)

		per_pdf_json = output_dir / pdf.stem / "result.json"
		per_pdf_json.parent.mkdir(parents=True, exist_ok=True)
		per_pdf_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Wrote: {per_pdf_json}")

	if args.json_file:
		json_path = Path(args.json_file)
		if not json_path.is_absolute():
			json_path = output_dir / json_path
		json_path.parent.mkdir(parents=True, exist_ok=True)

		payload: Any
		if len(all_results) == 1:
			payload = all_results[0]
		else:
			payload = {"results": all_results}

		json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"Wrote: {json_path}")


# ─── Configuration (edit these to run directly without CLI arguments) ───
INPUT_PATH = r".\AF_Berichte"  # PDF file or folder
OUTPUT_DIR = r".\outputs"  # Output directory
# ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
	main()

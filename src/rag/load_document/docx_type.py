from pathlib import Path
from typing import Dict, List, Tuple

from docx import Document as DocxDocument

from src.rag.load_document.base_document import BaseDocumentExtractor


class DOCXExtractor(BaseDocumentExtractor):
    """
    Trích xuất text và ảnh nhúng từ file DOCX.
    - Duyệt theo thứ tự xuất hiện của paragraph / table / image
    - Ảnh lấy từ relationships của document
    """

    def _extract_blocks(self, file_path: Path) -> List[Tuple[int, str, bytes | str]]:
        doc = DocxDocument(str(file_path))
        blocks: List[Tuple[int, str, bytes | str]] = []
        order = 0

        # Build a map: rId → image bytes from document relationships
        image_map: Dict[str, bytes] = {}
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_map[rel.rId] = rel.target_part.blob
                except Exception:
                    pass

        def _iter_body_elements():
            """Yield (element_type, element) in document order."""
            body = doc.element.body
            for child in body:
                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                yield tag, child

        for elem_tag, elem in _iter_body_elements():
            if elem_tag == "p":
                # Check for inline images
                drawing_elements = elem.findall(
                    ".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}inline"
                ) + elem.findall(
                    ".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}anchor"
                )
                for drawing in drawing_elements:
                    blip = drawing.find(
                        ".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
                    )
                    if blip is not None:
                        r_embed = blip.get(
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                        )
                        if r_embed and r_embed in image_map:
                            blocks.append((order, "image", image_map[r_embed]))
                            order += 1

                # Plain text
                text = elem.text or "".join(
                    run.text for run in elem.runs if run.text
                )
                if text.strip():
                    blocks.append((order, "text", text))
                    order += 1

            elif elem_tag == "tbl":
                # Extract table as structured text
                table_lines: List[str] = []
                rows = elem.findall(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"
                )
                for row in rows:
                    cells = row.findall(
                        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"
                    )
                    cell_texts = []
                    for cell in cells:
                        paras = cell.findall(
                            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
                        )
                        cell_text = " ".join(
                            "".join(
                                r.text or ""
                                for r in p.findall(
                                    ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
                                )
                            )
                            for p in paras
                        )
                        cell_texts.append(cell_text.strip())
                    table_lines.append(" | ".join(cell_texts))
                table_str = "\n".join(table_lines)
                if table_str.strip():
                    blocks.append((order, "text", table_str))
                    order += 1

        return blocks

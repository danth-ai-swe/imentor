import glob
import os
import time

import comtypes.client


def convert_single(word_app, docx_path: str, pdf_path: str):
    doc = None
    try:
        doc = word_app.Documents.Open(os.path.abspath(docx_path))
        time.sleep(1)
        doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)
    finally:
        if doc:
            doc.Close(False)


def batch_convert(input_dir: str, output_dir: str):
    docx_files = glob.glob(os.path.join(input_dir, "**", "*.docx"), recursive=True)

    if not docx_files:
        print("Không tìm thấy file .docx nào.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Tìm thấy {len(docx_files)} file...\n")

    for i, docx_path in enumerate(docx_files, 1):
        pdf_name = os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
        pdf_path = os.path.join(output_dir, pdf_name)

        word = None
        try:
            word = comtypes.client.CreateObject("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False

            convert_single(word, docx_path, pdf_path)
            print(f"[{i}/{len(docx_files)}] ✅ {os.path.basename(docx_path)}")

        except Exception as e:
            print(f"[{i}/{len(docx_files)}] ❌ Lỗi: {os.path.basename(docx_path)} — {e}")

        finally:
            if word:
                try:
                    word.Quit()
                except:
                    pass
            time.sleep(0.5)  # Tránh Word bị quá tải

# from ingestion.converters import pdf2md_marker, pdfloader

import os


def get_root_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_file_path(path):
    return os.path.join(get_root_path(), path)


if __name__ == "__main__":
    f_path = get_file_path("temp/agentx.pdf")
    # print("===== Converting PDF to Markdown =====")
    # md_path = pdf2md_marker.convert(file_path)
    # print(md_path, "saved")
    print(f_path)
    # pdfloader.load_pdf(f_path)


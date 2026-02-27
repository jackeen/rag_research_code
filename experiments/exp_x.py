from alpha import ingester
import os


if __name__ == '__main__':
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pdf_path = os.path.join(root_path, 'temp/agentx.pdf')
    ingester.ingest_pdf(pdf_path, "exp_1")

"""

https://docs.langchain.com/oss/python/integrations/document_loaders/pymupdf
"""

from langchain_community.document_loaders import PyMuPDFLoader


def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    for doc in docs:
        print("---page---")
        print(doc.page_content)



from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"D:\Gementis\real-estate-doc-intelligence\data\Agriculture.pdf")
documents = loader.load()

for i in range(50):
    print(i, documents[i].metadata["page"])

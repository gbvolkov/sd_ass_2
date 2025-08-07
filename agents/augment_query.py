from agents.retrievers.teamly_retriever import TeamlyRetriever_Glossary

glossary_retriever = TeamlyRetriever_Glossary(auth_data_store="./auth.json")

def get_terms_and_definitions(query: str) -> str:
    tnd_docs = glossary_retriever.invoke(query)
    return "\n\n".join(
        d.page_content.strip()
        for d in tnd_docs
        if getattr(d, "page_content", None)
    )

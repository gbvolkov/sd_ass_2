from langchain_core.tools import tool

@tool
def get_support_contact() -> str:
    """Returns list of technical support contacts."""
    return ["telegram: https://t.me/zerocoder_study_bot", "WhatsApp: +7(993)3775209"]

@tool
def get_customer_manager_contact(customer: str) -> str:
    """Returns list of customer manager contacts."""
    return ["Please find information on our site: [Zerocoder](https://zerocoder.ru/)"]


@tool
def get_discounts_and_actions(topic: str) -> str:
    """Returns information about current and upcoming discounts and actions."""

    return "В университете иногда проходят распродажи и выдаются гранты.\nПожалуйста, свяжитесь отделом по работе с клиентами, чтобы уточнить наличие скидок."

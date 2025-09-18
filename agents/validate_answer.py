from typing import Optional
from pydantic import BaseModel, Field
import logging
import config

#from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from agents.llm_utils import get_llm

class CheckResult(BaseModel):
    """
    Class containing result ()YES or NO) of a check and a reason whay the result is negative.
    """
    result: str = Field(default=None, description = "YES or NO")
    reason: Optional[str] = Field(default=None, description="In case result is NO, put here short explanation why answer is incorrect. Otherwise leave if empty.")

with open("prompts/check_answer_prompt.txt", encoding="utf-8") as f:
    prompt_txt = f.read()

prompt = PromptTemplate.from_template(prompt_txt)
validation_provider = config.VALIDATION_LLM
llm = get_llm(model="base", provider=validation_provider)
#= ChatOpenAI(model="gpt-5", temperature=0)
check_llm = llm.with_structured_output(CheckResult)
check_chain = prompt | check_llm

def vadildate_AI_answer(question: str, answer: str) -> CheckResult:
    try:
        return check_chain.invoke({"question": question, "answer": answer})
    except Exception as e:
        logging.error("Error occured at vadildate_AI_answer.\nException: {e}")
        raise e


if __name__ == "__main__":
    import os

    os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    qas = [
        {
            "question": "Как в CRM создать новую задачу на бэк-службу?",
            "answer": """
🔧 **Решение**

🧠 **Понимание проблемы**  
Вы хотите узнать, как в CRM создать новую задачу на бэк-службу.

💡 **Рекомендуемое решение**  
В CRM новые задачи на бэк-службу можно создавать следующим образом:
- Автоматически при запуске ЛЗ на согласование (в соответствии с карточкой ОБП и разделом Заявки).
- Из списка шаблонов задач на вкладке «Задачи в бэк-службу» в карточке клиента (раздел Клиенты).

То есть для создания новой задачи на бэк-службу в CRM зайдите в карточку клиента, далее в раздел задач и выберите из списка шаблонов соответствующую задачу, после чего она будет создана и назначена на бэк-службу.

ℹ️ **Дополнительная информация**  
• Задачи в бэк-службу часто связаны с заведением карточек контрагентов, запуском запросов и проверок в рамках бизнес-процессов лизинга.  
• Для удобства работы с задачами существуют специализированные карточки и шаблоны задач.  
• При необходимости задачи могут запускать бизнес-процессы и маршруты для дальнейшей обработки и согласования.

📚 **Релевантные записи**  
1. *Создание задач* — о способах создания задач в CRM и работе с шаблонами задач бэк-службы.  
2. *Как структурировать сделку до отправки на согласование по "Экспрессу"* — примеры задач в бэк-службу и рекомендации по созданию задач.  
3. *Обновление 20241128* — улучшения по задачам и бэк-службе для корректной работы и возврата задач на доработку.

❓ **Дополнительные вопросы**  
Также я мог бы помочь Вам ответить на следующие вопросы:  
1. Как настроить шаблоны задач в CRM?  
2. Как отслеживать статус задач в бэк-службе?  
3. Как автоматизировать задачи на бэк-службу через бизнес-процессы?
"""
    },
    {
        "question": "как оформить агентский договор",
        "answer": """
К сожалению, не удалось получить необходимые данные из базы знаний по оформлению агентского договора. Ниже кратко об основных моментах:

Для оформления агентского договора обычно нужно:  
- Стороны договора (принципал и агент).  
- Предмет договора (что агент будет делать для принципала).  
- Права и обязанности сторон.  
- Срок действия договора.  
- Условия оплаты агентского вознаграждения.  
- Ответственность сторон и порядок разрешения споров.  

Рекомендую обратиться к внутренним шаблонам и документам компании "Интерлизинг" или проконсультироваться с юридическим отделом для точного оформления.

Если хотите, могу продолжить поиск или помочь с другими вопросами."""}
    ]

    for qa in qas:
        result = vadildate_AI_answer(question=qa["question"], answer=qa["answer"])
        print(result)

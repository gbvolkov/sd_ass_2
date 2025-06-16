import requests
import json
import time

import config
model_uri=f'cls://{config.YA_FOLDER_ID}/yandexgpt-lite/latest'
header = {
    "Content-Type": "application/json",
    "Authorization": "Api-Key " + config.YA_API_KEY
}
api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/fewShotTextClassification"

classification_task = "Определи тип запроса пользователя к базе знаний компании."
samples = [
    {
        "text": "Как оформить коммандировку?",
        "label": "управление персоналом"
    },
    {
        "text": "Какие параметры нужно указывать в лизинговой заявке?",
        "label": "продажи и продукты"
    },
    {
        "text": "не работает предрешение, что делать?",
        "label": "техническая поддержка"
    },
    {
        "text": "Какие есть субсидии?",
        "label": "продажи и продукты"
    },
    ]
lables = {
    "продажи и продукты": "sm_agent",
    "управление персоналом": "default_agent",
    "техническая поддержка": "sd_agent",
    "другое": "default_agent"
    }


def classify_request(request: str) -> str:
    prompt = {
        "modelUri": model_uri,
        "text": request,
        "task_description": classification_task,
        "labels": list(lables.keys()),
        "samples": samples
    }
    try:
        response = requests.post(api_url, headers=header, json=prompt)
        response.raise_for_status()
        result = json.loads(response.text)
        predictions = result["predictions"]
        predictions.sort(key = lambda x: x["confidence"], reverse=True)
        defined_class = predictions[0]["label"]
        return lables[defined_class]
    except:
        return "default_agent"

if __name__ == "__main__":
    user_queries =  [
        "Какой порядок выхода в отпуск?",
        "Какие виды субсидий МПТ существуют?",
        "Как настроить калькулятор для коммерческого менеджера?",
        "Какая погода в Бомбее?",
    ]
    for query in user_queries:
        query_class = classify_request(query)
        print(f"{query}:{query_class}")
        time.sleep(1)






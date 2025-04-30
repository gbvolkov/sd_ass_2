#!/usr/bin/env python3
# coding: utf-8
"""
tg_chat2rag.py  –  извлечение Q&A из Telegram‑чата Интенсива Zerocoder

(1) Определяет куратора по фразе «Ваш куратор» в приветственном сообщении
(2) Находит все его ответы (reply‑сообщения) и фиксирует,
    кому и на какое сообщение он ответил
(3) Собирает «вопросы»  –  сообщения с хеш‑тегом  #вопроскуратору
    + все сообщения, на которые ответил куратор
(4) Строит исчерпывающий список пар Question/Answer для RAG
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import glob

import operator

def load_chat(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_text(msg: Dict[str, Any]) -> str:
    """
    Забирает plain‑текст из поля 'text' (строка ИЛИ массив сегментов).
    """
    txt = msg.get("text")
    if isinstance(txt, str):
        return txt.strip()
    elif isinstance(txt, list):
        chunks = []
        for seg in txt:
            if isinstance(seg, str):
                chunks.append(seg)
            elif isinstance(seg, dict):
                chunks.append(seg.get("text", ""))
        return "".join(chunks).strip()
    return ""


# --------------------------------------------------------------------------- #
# 1) Поиск куратора
# --------------------------------------------------------------------------- #
def find_curators_old(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Возвращает {user_id: display_name}
    Ищем фразу 'Ваш куратор' в приветственном посте (есть @username).
    """
    curators = {}
    patt = re.compile(r"Ваш куратор.*?@(\w+)", re.I | re.S)

    for m in messages:
        txt = flatten_text(m)
        mobj = patt.search(txt)
        if mobj:
            username = mobj.group(1)
            # ищем первое сообщение, где from.username совпадает
            for mm in messages:
                if mm.get("from") and username in mm.get("from"):
                    curators[mm["from_id"]] = mm["from"]
                    break
            # можно поддержать несколько кураторов
    return curators

def find_curators(messages):
    """
    Возвращает {user_id: display_name}

    1. Ищем username куратора в приветственном сообщении
    2. Считаем, у каких from_id больше всего сообщений с #ответ
       – обычно это и есть куратор.
    3. Если username встречается в display‑name, оставляем только таких авторов.
    """
    # 1. username из приветствия
    #welcome_pat = re.compile(r"Ваш куратор[^@]*@(\w+)", re.I)
    #username = None
    #for m in messages:
    #    mtxt = flatten_text(m)
    #    mobj = welcome_pat.search(mtxt)
    #    if mobj:
    #        username = mobj.group(1).lower()
    #        break
    #username = None

    # 2. кто чаще всего использует #ответ
    counter = defaultdict(int)
    for m in messages:
        if "#ответ" in flatten_text(m).lower():
            counter[m["from"]] += 1

    if not counter:
        raise RuntimeError("В чате нет сообщений с тегом #ответ – не могу найти куратора")

    max_hits = max(counter.values())
    candidate_ids = [uid for uid, n in counter.items() if n == max_hits]
    # 3. фильтруем по совпадению username -> display‑name
    curators = {}
    for m in messages:
        uid = m.get("from", "")
        if uid in candidate_ids:
            display = m.get("from", "")
            curators[uid] = display

    if not curators:
        raise RuntimeError("username найден, но автор совпадающих #ответ‑сообщений не обнаружен")

    return curators


# --------------------------------------------------------------------------- #
# 2) Извлечение ответов куратора
# --------------------------------------------------------------------------- #
def extract_answers(messages, curators):
    answers = []
    by_id = {m["id"]: m for m in messages}

    for m in messages:
        if m.get("from") in curators and m.get("reply_to_message_id"):
            ans = {
                "answer_id": m["id"],
                "answer_text": flatten_text(m),
                "answer_ts": m["date"],
                "curator_name": curators[m["from"]],
                "question_id": m["reply_to_message_id"],
            }
            qmsg = by_id.get(ans["question_id"])
            if qmsg:
                ans["question_text"] = flatten_text(qmsg)
                ans["question_author"] = qmsg.get("from", "")
            else:
                ans["question_text"] = ""
                ans["question_author"] = ""
            answers.append(ans)
    return answers


# --------------------------------------------------------------------------- #
# 3) Поиск всех «вопросов»
# --------------------------------------------------------------------------- #
def collect_questions(messages):
    questions = {}
    for m in messages:
        txt = flatten_text(m).lower()
        if "#вопроскуратору" in txt:
            questions[m["id"]] = {
                "question_id": m["id"],
                "question_text": flatten_text(m).replace("#вопроскуратору", "").strip(),
                "question_author": m.get("from", ""),
            }
    return questions


# --------------------------------------------------------------------------- #
# 4) Сборка финальной Q&A‑коллекции
# --------------------------------------------------------------------------- #
def build_qa(answers, questions):
    """
    answers  – список словарей (см. extract_answers)
    questions – dict {id: {...}}
    Возвращаем list[dict] с ключами: question / answer / meta
    """
    qa_pairs = []

    used_qids = set()
    for ans in answers:
        qid = ans["question_id"]
        used_qids.add(qid)
        qa_pairs.append(
            {
                "question": ans["question_text"]
                or questions.get(qid, {}).get("question_text", ""),
                "answer": re.sub(r"#ответ", "", ans["answer_text"], flags=re.I).strip(),
                "meta": {
                    "question_id": qid,
                    "answer_id": ans["answer_id"],
                    "question_author": ans["question_author"],
                    "answer_author": ans["curator_name"],
                    "answer_ts": ans["answer_ts"],
                },
            }
        )

    # вопросы, на которые куратор ещё не ответил
    for qid, qrec in questions.items():
        if qid not in used_qids:
            qa_pairs.append(
                {
                    "question": qrec["question_text"],
                    "answer": "",  # нет ответа пока
                    "meta": {
                        "question_id": qid,
                        "question_author": qrec["question_author"],
                    },
                }
            )

    return qa_pairs


def hashtags(text: str) -> list[str]:
    """возвращает список всех #тегов из строки"""
    return re.findall(r"#\w+", text, flags=re.I)

def simplified_qa(json_path):
    chat = load_chat(json_path)
    messages = chat["messages"]

    #filtered = [m for m in msgs if m.get('from')] 
    #messages = sorted(filtered, key=operator.itemgetter('from', 'id'))

    curators = find_curators(messages)
    curatorstr = ";".join([k for k, _ in curators.items()])
    inistr = f"{json_path}======================= Куратор:{curatorstr}\n"
    msgs = []
    for msg in messages:
        text = flatten_text(msg)
        author=msg.get("from", "UNKNOWN")
        hashtag = hashtags(text)
        if author in curators:
            author = author + " (роль:куратор)"
        else:
            author = author + " (роль:студент)"
        msgs.append(f"Author:{author}\n{text}\nHashtags:{";".join(hashtag) if isinstance(hashtag, list) else hashtag}\n\n")
    return inistr + "=======================\n".join(msgs)


def convert_json_to_csv(input_file, output_file):
    import json
    import csv
    from pathlib import Path

    # ---- settings -------------------------------------------------------------
    INFILE  = Path(input_file)   # <‑‑ put your JSON file name here
    OUTFILE = Path(output_file)    # <‑‑ desired CSV name
    # ---------------------------------------------------------------------------

    with INFILE.open(encoding="utf‑8") as f:
        data = json.load(f)        # list[dict]

    # Define the CSV columns once, in the order you want
    fieldnames = [
        "question",
        "answer",
        "question_id",
        "answer_id",
        "question_author",
        "answer_author",
        "answer_ts",
    ]

    with OUTFILE.open("w", newline="", encoding="utf‑8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in data:
            meta = item.get("meta", {})
            writer.writerow(
                {
                    "question":        item.get("question", ""),
                    "answer":          item.get("answer", ""),
                    "question_id":     meta.get("question_id", ""),
                    "answer_id":       meta.get("answer_id", ""),
                    "question_author": meta.get("question_author", ""),
                    "answer_author":   meta.get("answer_author", ""),
                    "answer_ts":       meta.get("answer_ts", ""),
                }
            )

    print(f"✔ CSV written to {OUTFILE.resolve()}")

# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":


    import argparse, pprint, datetime as dt

    #ap = argparse.ArgumentParser(description="Extract RAG Q&A from Telegram chat JSON")
    #ap.add_argument("json_path", help="Path to result.json exported by tdlib")
    #ap.add_argument(
    #    "-o",
    #    "--out",
    #    help="Where to save QA as JSON (default: print to stdout)",
    #    default=None,
    #)
    #args = ap.parse_args()


    out = "data/qa.txt"
    qas = []
    for json_path in glob.glob("data/chats/native/*.json"):
    #json_path = "data/chats/native/result (1).json"
        try:
            qas.append(simplified_qa(json_path))
        except Exception as e: 
            print(f"Failed to parse {json_path}: {e}")

    if out:
        Path(out).write_text(f"{json_path}======================================\n".join(qas), encoding="utf-8")
        print(f"Saved {len(qas)} QA messages → {out}")
    else:
        pprint.pp(qas[:2])  # печатаем первые 10 для проверки
else:
    import argparse, pprint, datetime as dt

    #ap = argparse.ArgumentParser(description="Extract RAG Q&A from Telegram chat JSON")
    #ap.add_argument("json_path", help="Path to result.json exported by tdlib")
    #ap.add_argument(
    #    "-o",
    #    "--out",
    #    help="Where to save QA as JSON (default: print to stdout)",
    #    default=None,
    #)
    #args = ap.parse_args()


    out = "data/qa.json"
    qas = []
    for json_path in glob.glob("data/chats/native/*.json"):
    #json_path = "data/chats/native/result (1).json"
        try:
            chat = load_chat(json_path)
            msgs = chat["messages"]
            filtered = [m for m in msgs if m.get('from')] 
            messages = sorted(filtered, key=operator.itemgetter('from', 'id'))
            curators = find_curators(messages)
            if not curators:
                raise RuntimeError("Куратор не найден: проверьте текст приветствия")

            answers = extract_answers(messages, curators)
            questions = collect_questions(messages)
            qa = build_qa(answers, questions)
            qas.extend(qa)
        except Exception as e: 
            print(f"Failed to parse {json_path}: {e}")

    if out:
        Path(out).write_text(json.dumps(qas, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {len(qas)} Q&A pairs → {out}")
    else:
        pprint.pp(qa[:10])  # печатаем первые 10 для проверки

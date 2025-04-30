import os
import re
import json
import glob
from bs4 import BeautifulSoup, NavigableString

def html_to_markdown(element):
    """Recursively converts an HTML element to markdown text."""
    markdown = ""
    for child in element.children:
        if isinstance(child, NavigableString):
            markdown += child.string
        elif child.name == 'br':
            markdown += "\n"
        elif child.name == 'strong':
            # Convert <strong> to markdown bold
            markdown += "**" + html_to_markdown(child) + "**"
        elif child.name == 'a':
            href = child.get("href", "")
            text = html_to_markdown(child)
            if href:
                markdown += f"[{text}]({href})"
            else:
                markdown += text
        else:
            markdown += html_to_markdown(child)
    return markdown

def parse_telegram_chat(html_content):
    """Parses a Telegram chat HTML export and returns a list of messages in structured JSON format.
    Groups parts by messageID. For each message, the author is taken from the first encountered part.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    messages_by_id = {}
    
    # Process all message divs with class "message default"
    for message_div in soup.find_all("div", class_=re.compile(r"message default")):
        # Extract messageID from the div id (e.g. "message186" -> 186)
        div_id = message_div.get("id", "")
        m_id = re.search(r"message-?(\d+)", div_id)
        if not m_id:
            continue
        message_id = int(m_id.group(1))
        
        body_div = message_div.find("div", class_="body")
        if not body_div:
            continue
        
        # Extract author (if available)
        from_name_div = body_div.find("div", class_="from_name")
        author = from_name_div.get_text(strip=True) if from_name_div else None
        
        # Extract datetime from the date div's title attribute
        date_div = body_div.find("div", class_="pull_right date details")
        datetime_str = date_div.get("title", "").strip() if date_div else None
        
        # Extract content and convert to markdown
        text_div = body_div.find("div", class_="text")
        content = html_to_markdown(text_div) if text_div else ""
        
        # Extract hashtags from content (words starting with #)
        hashtags = re.findall(r"#\w+", content, flags=re.UNICODE)
        
        # Check for reply block and extract parent message ID
        parent = None
        reply_div = body_div.find("div", class_="reply_to details")
        if reply_div:
            a_tag = reply_div.find("a")
            if a_tag and a_tag.get("href"):
                m_parent = re.search(r"go_to_message(\d+)", a_tag["href"])
                if m_parent:
                    parent = int(m_parent.group(1))
        
        # Build the part dictionary
        part = {
            "content": content.strip(),
            "datetime": datetime_str,
            "parent": parent,
            "hashtags": hashtags
        }
        
        # Group parts by message_id. Use the first discovered author.
        if message_id not in messages_by_id:
            messages_by_id[message_id] = {
                "messageID": message_id,
                "author": author,
                "parts": [part]
            }
        else:
            # If author not set yet, use the current one if available.
            if messages_by_id[message_id]["author"] is None and author is not None:
                messages_by_id[message_id]["author"] = author
            messages_by_id[message_id]["parts"].append(part)
    
    # Return the grouped messages as a list
    return list(messages_by_id.values())

if __name__ == "__main__":
    # Iterate over all HTML files in the current folder
    for input_file in glob.glob("./data/data/*.html"):
        with open(input_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        parsed_messages = parse_telegram_chat(html_content)
        
        # Convert parsed messages to a formatted JSON string
        json_output = json.dumps(parsed_messages, ensure_ascii=False, indent=2)
        
        # Save JSON to a file with the same base name
        base_name, _ = os.path.splitext(input_file)
        output_file = base_name + ".json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_output)
        
        print(f"JSON saved to {output_file}")

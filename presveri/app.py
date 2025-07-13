from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import filetype
import os
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import json
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'])

# Helper function to extract text from a file (image or PDF)
def extract_text_from_file(file):
    text_output = ""

    # Identify file type
    kind = filetype.guess(file.read(261))
    file.seek(0)  # Reset file pointer

    if kind and kind.mime.startswith("image"):
        img = Image.open(file.stream).convert('RGB')
        result = reader.readtext(np.array(img), detail=0)
        text_output = " ".join(str(r) for r in result)

    elif (kind and "pdf" in kind.mime) or file.filename.lower().endswith(".pdf"):
        temp_path = "temp_prescription.pdf"
        with open(temp_path, 'wb') as f:
            f.write(file.read())

        images = convert_from_path(temp_path, dpi=300)
        for img in images:
            img_np = np.array(img.convert('RGB'))
            result = reader.readtext(img_np, detail=0)
            text_output += " " + " ".join(str(r) for r in result)

        os.remove(temp_path)
    else:
        raise ValueError("Unsupported file type")

    return text_output.strip()

# Groq LLM verification logic
def verify_with_groq(ocr_text, desired_items):
    desired_names = [item["name"] for item in desired_items]

    prompt = f"""
You are a medical assistant. The following text was extracted from a handwritten prescription:

\"\"\"{ocr_text}\"\"\"

Check if any of these medicines are mentioned:
{', '.join(desired_names)}

Return a JSON list of the medicine names that are clearly mentioned in the text. If none are mentioned, return an empty list.
""".strip()

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",  # Or: mixtral-8x7b-32768, gemma-7b-it
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = httpx.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise ValueError(f"Groq API error: {response.text}")

    result = response.json()

    try:
        content = result["choices"][0]["message"]["content"]

        # Try to extract JSON list manually using regex
        match = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("Groq response did not contain valid JSON list:\n" + content)
    except KeyError:
        raise ValueError("Groq API response missing 'choices':\n" + json.dumps(result, indent=2))
    except Exception as e:
        raise ValueError(f"Groq LLM response parsing failed: {e}")

@app.route("/verify-prescription", methods=["POST"])
def verify_prescription():
    if 'files' not in request.files:
        return jsonify({"error": "Missing 'files' in request"}), 400
    if 'desired_items' not in request.form:
        return jsonify({"error": "Missing 'desired_items' in request"}), 400

    try:
        files = request.files.getlist('files')
        desired_items = json.loads(request.form['desired_items'])

        combined_text = ""
        for file in files:
            combined_text += " " + extract_text_from_file(file)

        # LLM-assisted verification
        matched_items = verify_with_groq(combined_text, desired_items)

        matched_lower = [m.lower() for m in matched_items]
        unmatched_items = [item for item in desired_items if item["name"].lower() not in matched_lower]
        prescribed_items = [{"medication_name": name.title()} for name in matched_items]

        response = {
            "ocr_text": combined_text,
            "desired_items": desired_items,
            "verification_result": {
                "is_valid_order": bool(matched_items) ,
                
                "identified_prescribed_items": prescribed_items,
                "verification_details": (
                    "All medications found in prescription"
                    if len(matched_items) == len(desired_items)
                    else "Partial match found" if matched_items else "No matches found"
                ),
                "matched_items": [name.title() for name in matched_items],
                "unmatched_desired_items": unmatched_items,
                "additional_notes": (
                    "All requested medications are valid"
                    if not unmatched_items
                    else "Some items missing"
                )
            },
            "message": "Verification complete"
        }

        return jsonify(response)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format in 'desired_items'"}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=6000)

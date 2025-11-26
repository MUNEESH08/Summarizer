from flask import Flask, render_template, request
import os
import cohere

app = Flask(__name__)

# Load API key from environment
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = request.form.get("resume_text", "")

    prompt = f"""
    Extract all technical and soft skills mentioned in the following resume.
    Then rank them by how strongly the resume indicates confidence or experience.
    Return ONLY the top 2 skills in a JSON array like ["Skill1", "Skill2"].

    Resume text:
    {resume_text}
    """

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=100,
        temperature=0.2
    )

    output = response.generations[0].text.strip()

    return render_template("result.html", skills=output)

if __name__ == "__main__":
    app.run(debug=True)

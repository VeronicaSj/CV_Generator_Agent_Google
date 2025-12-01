Enable virtual enviroment

 - python -m venv venv
 - source venv/bin/activate  # On Linux/macOS
 - .\venv\Scripts\activate  # On Windows

Install Dependencies: You will need the ADK, the Google Generative AI SDK (for Gemini), a local vector store, and a PDF generation library.

 - pip install google-genai google-adk pydantic chromadb reportlab

Run the agent:

 - adk run CV_generator

Say hi to start.

# cv_tool.py

import os
import json
import chromadb
from google.adk.tools import AgentTool
from google import genai
from pydantic import BaseModel, Field
from typing import List
from chromadb.utils import embedding_functions
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from chromadb.utils import embedding_functions
from typing import List

from google import genai

client = genai.Client()

embedding = client.models.embed_content(
    model="models/text-embedding-004",
    contents="some text"
)

# --- Pydantic Schemas for Structured Output ---

class ExtractedRequirements(BaseModel):
    """Structured output for the Job Analyzer Module."""
    required_skills: List[str] = Field(description="Essential skills and technologies mentioned.")
    preferred_skills: List[str] = Field(description="Skills that are a plus, but not mandatory.")
    responsibilities_keywords: List[str] = Field(description="Key verbs and nouns from the responsibilities section.")
    seniority_level: str = Field(description="Inferred level (e.g., 'Junior', 'Mid-level', 'Senior', 'Lead').")

# --- Core Tool Logic ---

class CVGeneratorTool(AgentTool):
    """
    Handles the entire CV generation workflow using Gemini and ChromaDB for RAG.
    """
    def __init__(self, agent, model_name: str = 'gemini-2.5-pro'):
        super().__init__(agent=agent)
        self.client = genai.Client()
        self.model_name = model_name
        
        self.db_client = chromadb.PersistentClient(path="./cv_profiles_db")

        class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self):
                self.client = genai.Client()

            def extract_vector(self, embedding):
                """
                embedding can be:
                - [ContentEmbedding(...)]
                - ContentEmbedding(...)
                - a list of floats
                """

                # Case A: embedding is a single-item list like [ContentEmbedding]
                if isinstance(embedding, list) and len(embedding) == 1:
                    embedding = embedding[0]

                # Case B: ContentEmbedding object
                if hasattr(embedding, "values"):
                    return embedding.values

                # Case C: already a list of floats (fallback)
                if isinstance(embedding, list) and all(isinstance(x, float) for x in embedding):
                    return embedding

                raise ValueError(f"Unknown embedding format: {embedding}")

            def __call__(self, texts: List[str]) -> List[List[float]]:
                vectors = []

                for text in texts:
                    response = self.client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=text
                    )

                    raw = response.embeddings

                    # FULLY unwrap to get float[]
                    vector = self.extract_vector(raw)

                    vectors.append(vector)

                return vectors
    
        # Initialize the collection (stores the actual vectorized data)
        self.profile_collection = self.db_client.get_or_create_collection(
            name="user_work_experience",
            embedding_function=GeminiEmbeddingFunction()
        )
        
        # Static user profile data (for simple list-based data)
        self.user_profile = {
            "name": "Alice Smith",
            "education": "M.S. Computer Science, 2018",
            "technical_skills_list": ["Python", "React", "TypeScript", "AWS", "SQL", "Microservices", "CI/CD", "Docker", "Kubernetes"],
            "soft_skills_list": ["Leadership", "Mentoring", "Communication", "Problem-Solving"]
        }
        
        # Populate the database on first run
        self._initialize_profile_data()

    def _initialize_profile_data(self):
        """Adds initial work experience to the vector store if it's empty."""
        initial_experiences = [
            "Senior Software Engineer at TechCorp (2020-2025). Led development of a high-traffic microservice. Optimized database queries for a 40% performance gain. Mentored junior staff.",
            "Junior Developer at WebSolutions (2018-2020). Built front-end components using React and TypeScript. Managed CI/CD pipelines."
        ]
        
        if self.profile_collection.count() == 0:
            print("INFO: Initializing ChromaDB with example work history for user_work_experience.")
            self.profile_collection.add(
                documents=initial_experiences,
                ids=[f"work_exp_{i}" for i in range(len(initial_experiences))]
            )

    def analyze_job_posting(self, job_posting: str) -> ExtractedRequirements:
        """Job Analyzer Module: Extracts structured requirements."""
        
        analysis_prompt = (
            "Analyze the following job posting and extract the structured information. "
            "Focus on high-value, ATS-critical keywords. "
            f"Job Posting:\n---\n{job_posting}\n---"
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[analysis_prompt],
            config={"response_mime_type": "application/json",
                    "response_schema": ExtractedRequirements,
            }
        )
        
        return ExtractedRequirements.model_validate_json(response.text)

    def get_relevant_experience(self, job_reqs: ExtractedRequirements) -> List[str]:
        """Retrieves the most relevant work experience from ChromaDB (RAG)."""
        
        # Combine high-value keywords for the query to ensure semantic relevance
        query_text = " ".join(job_reqs.required_skills + job_reqs.responsibilities_keywords)
        
        # Query ChromaDB for the top 3 most relevant work experiences (documents)
        results = self.profile_collection.query(
            query_texts=[query_text],
            n_results=3,
            include=['documents']
        )
        
        if results and results['documents']:
            return results['documents'][0]
        return []

    def generate_ats_optimized_section(self, section_type: str, job_requirements: ExtractedRequirements) -> str:
        """Matching & Scoring Engine: Generates an ATS-optimized summary."""

        if section_type == "Work Experience":
            original_content = "\n".join(self.get_relevant_experience(job_requirements))
        elif section_type == "Skills":
            original_content = f"Technical: {', '.join(self.user_profile['technical_skills_list'])}\nSoft: {', '.join(self.user_profile['soft_skills_list'])}"
        else:
            return f"Error: Unknown section type {section_type}"

        # This prompt is the core of the "Keyword Optimization Engine"
        matching_prompt = (
            f"You are the ATS Optimization Engine. Your goal is to rewrite the user's {section_type} "
            "section to maximize its compatibility score with the job requirements. "
            "Use the **exact keywords** from the job requirements naturally in the user's existing text. "
            "Be concise and professional, using strong action verbs."
            f"\n\n--- Job Requirements ---\nRequired Skills: {job_requirements.required_skills}"
            f"\nPreferred Skills: {job_requirements.preferred_skills}"
            f"\n\n--- User's Original {section_type} (Relevant Only) ---\n"
            f"{original_content}"
        )
            
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[matching_prompt]
        )
        return response.text.strip()

    def generate_pdf_cv(self, optimized_content: dict) -> str:
        """CV Rendering Service: Produces a simple PDF (using reportlab)."""
        
        filename = f"CV_Optimized_for_{optimized_content['job_title'].replace(' ', '_')}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        c.drawString(72, 750, f"CV for {self.user_profile['name']}")
        c.drawString(72, 735, f"Optimized for: {optimized_content['job_title']}")
        c.line(72, 730, 550, 730)
        
        y_position = 710
        
        sections = ["Work Experience", "Skills Summary"]
        for title in sections:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, title.upper())
            y_position -= 15
            
            content = optimized_content.get(title, "N/A")
            c.setFont("Helvetica", 10)
            
            # Simple line wrapping
            for line in content.split('\n'):
                if y_position < 50: 
                    c.showPage()
                    y_position = 750
                c.drawString(82, y_position, line)
                y_position -= 15
                
            y_position -= 10 # Extra space after section
        
        c.save()
        return filename

    # --- Tool for Continuous Profile Improvement ---
    
    def update_profile_with_new_experience(self, new_experience_description: str) -> str:
        """
        Agent action to update the user's long-term profile memory (ChromaDB).
        """
        # Generate a unique ID
        new_id = f"work_exp_{self.profile_collection.count() + 1}"
        
        # Add the new experience to ChromaDB, which automatically embeds it
        self.profile_collection.add(
            documents=[new_experience_description],
            ids=[new_id]
        )
        
        return f"Profile successfully updated! The new experience ('{new_experience_description[:30]}...') is now part of your permanent memory."


    # --- The main ADK entry point method ---

    def generate_optimized_cv(self, job_posting: str, job_title: str) -> str:
        """
        The main function that orchestrates the CV generation process.

        :param job_posting: The full text of the job description.
        :param job_title: The title of the job (for PDF naming).
        :return: A success message with the filename of the generated CV.
        """
        try:
            # 1. Job Analyzer Module
            job_reqs = self.analyze_job_posting(job_posting)

            # 2. Matching & Scoring Engine
            optimized_experience = self.generate_ats_optimized_section(
                "Work Experience", job_reqs
            )
            optimized_skills = self.generate_ats_optimized_section(
                "Skills", job_reqs
            )
            
            # 3. Gap Analysis (for Continuous Profile Improvement)
            user_skill_set = set(self.user_profile['technical_skills_list'] + self.user_profile['soft_skills_list'])
            missing_skills = [
                skill for skill in job_reqs.required_skills 
                if skill.lower() not in [s.lower() for s in user_skill_set]
            ]
            
            gap_report = ""
            if missing_skills:
                gap_report = f"**Gap Alert:** The job requires skills you haven't listed: **{', '.join(missing_skills)}**. Would you like to add a past experience that covers one of these?"

            # 4. CV Rendering Service
            optimized_content = {
                "job_title": job_title,
                "Work Experience": optimized_experience,
                "Skills Summary": optimized_skills,
            }

            pdf_filename = self.generate_pdf_cv(optimized_content)

            return (
                f"**CV Generated!** Your highly optimized CV for the **{job_title}** role has been generated."
                f"\nFile: **{pdf_filename}**."
                f"\n\n--- Next Steps ---\n{gap_report}"
            )

        except Exception as e:
            return f"An error occurred during CV generation: {e}"


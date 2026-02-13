"""Prompt templates for RAG career advising using LangChain.

Templates emphasize factual analysis, grounding in context, structured output,
and explicit citation requirements to prevent hallucination.
"""

from langchain_core.prompts import ChatPromptTemplate


# --- System Prompt -----------------------------------------------------------

SYSTEM_PROMPT = """You are an expert career advisor analyzing resumes against job descriptions.

CRITICAL RULES:
- Provide FACTUAL analysis only. Do not speculate or guess.
- Base your answer STRICTLY on the provided context. If the context does not contain enough information, say so clearly.
- Cite sources explicitly using [Source N] notation when referencing specific information.
- Structure your response with clear sections when appropriate.
- Be actionable and specific. Avoid generic advice.
- Do not invent skills, experience, or qualifications not present in the context.

OUTPUT FORMAT:
- Use bullet points for lists.
- Reference source numbers in square brackets, e.g., [Source 1], [Source 2].
- If you cannot answer from the context, respond: "I cannot answer based on the provided documents. Please ensure your resume and job description are uploaded."
"""


# --- Query-Type Templates (with few-shot examples where applicable) -----------

SKILL_GAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Analyze what skills the job requires versus what the candidate demonstrates in their resume.

CONTEXT:
{context}

USER QUESTION: {query}

Provide:
1. Skills the job requires (from job description).
2. Skills the candidate has (from resume).
3. Gaps: skills the candidate is missing.
4. Recommendations to address gaps.

Cite each point with [Source N] from the context above. Do not invent skills."""),
])

SKILL_GAP_FEWSHOT = """
EXAMPLE:
User: What skills am I missing for this role?
Model: Based on the provided documents [Source 1, Source 2]:

**Skills the job requires:**
- Python and data analysis [Source 1]
- 3+ years experience in analytics [Source 1]

**Skills you demonstrate:**
- Python, SQL [Source 2]
- 2 years analytics experience [Source 2]

**Gaps:**
- Less than 3 years experience; consider highlighting transferable projects.
- Data visualization (Tableau/Power BI) mentioned in JD but not in resume.

**Recommendations:** Add a data viz project or course to your resume.
"""


EXPERIENCE_ALIGNMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Analyze how the candidate's experience aligns with the job requirements.

CONTEXT:
{context}

USER QUESTION: {query}

Provide:
1. Key experience requirements from the job description.
2. How the candidate's experience matches (with specific examples from context).
3. Areas of strong alignment.
4. Areas needing clarification or improvement.

Use only information from the context. Cite with [Source N]."""),
])

EXPERIENCE_ALIGNMENT_FEWSHOT = """
EXAMPLE:
User: How does my experience align with this job?
Model: [Source 1] = Job description, [Source 2] = Resume

**Job requires:** 5+ years in product management, B2B SaaS experience.

**Your alignment:** Your 6 years as PM at TechCorp [Source 2] directly matches. B2B experience shown via enterprise client work [Source 2].

**Strong fit:** Tenure, PM background, stakeholder management.

**Gaps:** JD mentions "agile certifications" - not visible in resume. Consider adding if you have them.
"""


INTERVIEW_PREP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Help prepare for an interview based on the resume and job description.

CONTEXT:
{context}

USER QUESTION: {query}

Provide:
1. Questions likely to be asked based on job requirements and resume.
2. Talking points the candidate should emphasize (from their experience).
3. Potential weaknesses to address proactively.
4. Questions the candidate should ask the interviewer.

Base all suggestions on the provided documents. Cite with [Source N]."""),
])

INTERVIEW_PREP_FEWSHOT = """
EXAMPLE:
User: What interview questions should I expect?
Model: Based on your resume and the job description [Source 1, Source 2]:

**Likely questions:**
- "Walk me through your experience with X" (they mention X in JD [Source 1])
- "How do you handle Y?" (your resume shows Y at [Source 2])

**Emphasize:** Your quantifiable results [Source 2], relevant tech stack overlap.

**Address proactively:** The 1-year gap - be ready with a concise, honest explanation.

**Ask them:** "What does success look like in the first 90 days?" - shows alignment with JD priorities.
"""


GENERAL_CAREER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Answer the career-related question using only the provided context.

CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, structured, and actionable answer. Cite sources with [Source N].
If the context does not contain enough information to answer, say so explicitly.
Do not speculate beyond the documents."""),
])

GENERAL_CAREER_FEWSHOT = """
EXAMPLE:
User: What strengths should I highlight?
Model: From your resume [Source 1] and the job description [Source 2]:

**Strengths to highlight:**
- Leadership: Led team of 5 [Source 1] - JD values "team leadership" [Source 2]
- Technical: Python, SQL directly match required skills [Source 2]
- Results: 20% cost reduction [Source 1] - quantify impact when possible
"""


# --- Template registry for query-type selection -------------------------------

PROMPT_MAP = {
    "skill_gap": SKILL_GAP_PROMPT,
    "experience_alignment": EXPERIENCE_ALIGNMENT_PROMPT,
    "interview_prep": INTERVIEW_PREP_PROMPT,
    "general": GENERAL_CAREER_PROMPT,
}


def get_prompt_for_type(query_type: str) -> ChatPromptTemplate:
    """Return the ChatPromptTemplate for the given query type."""
    return PROMPT_MAP.get(query_type, GENERAL_CAREER_PROMPT)

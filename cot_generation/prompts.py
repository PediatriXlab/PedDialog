#!/usr/bin/env python3
"""
Prompt templates for CoT reasoning annotation generation.

Five physician response types (A-E), each with a type-specific reasoning structure.
See paper Methods: Reasoning Annotation Process for details.
"""

# Response type definitions
RESPONSE_TYPES = {
    "A": "Diagnostic reasoning",
    "B": "Treatment recommendation",
    "C": "Clinical inquiry",
    "D": "Health guidance",
    "E": "Follow-up recommendation",
}

# Classification prompt
CLASSIFY_PROMPT = """Classify the following physician response in a pediatric consultation.

# Dialog history
{dialog_history}

# Current physician response
{doctor_response}

# Types
A. Diagnostic reasoning: diagnosis conclusions, disease judgments
B. Treatment recommendation: medication advice, treatment plans
C. Clinical inquiry: follow-up questions about symptoms, history
D. Health guidance: care advice, dietary guidance, precautions
E. Follow-up recommendation: recheck timing, follow-up arrangements

Reply with a single letter (A/B/C/D/E)."""


# Type-specific CoT generation prompts
TYPE_PROMPTS = {
    "A": """You are a senior pediatrician at a tertiary children's hospital.

# Dialog history (up to current turn)
{dialog_history}

# Current physician response (Type A: Diagnostic reasoning)
{doctor_response}

Generate the physician's reasoning annotation covering:
1. Information completeness: what is known, what remains unknown
2. Symptom analysis: key features, duration, progression
3. Diagnostic reasoning: most likely diagnoses given available information
4. Differential diagnosis: conditions to consider or exclude
5. Guideline basis: relevant clinical guidelines

Output JSON: {{"inner_thinking": {{...}}}}
Only output the JSON object.""",

    "B": """You are a senior pediatrician at a tertiary children's hospital.

# Dialog history (up to current turn)
{dialog_history}

# Current physician response (Type B: Treatment recommendation)
{doctor_response}

Generate the physician's reasoning annotation covering:
1. Severity assessment: current condition severity and limitations of remote assessment
2. Treatment goals: expected outcomes
3. Treatment choice: rationale for medication or therapy selection
4. Safety considerations: side effects, monitoring needs, warning signs
5. Patient education: how to communicate instructions clearly

Output JSON: {{"inner_thinking": {{...}}}}
Only output the JSON object.""",

    "C": """You are a senior pediatrician at a tertiary children's hospital.

# Dialog history (up to current turn)
{dialog_history}

# Current physician response (Type C: Clinical inquiry)
{doctor_response}

Generate the physician's reasoning annotation covering:
1. Inquiry purpose: why this information is needed
2. Information completeness: what can and cannot be assessed remotely
3. Diagnostic value: how this information narrows the differential
4. Next steps: planned actions depending on the answer

Output JSON: {{"inner_thinking": {{...}}}}
Only output the JSON object.""",

    "D": """You are a senior pediatrician at a tertiary children's hospital.

# Dialog history (up to current turn)
{dialog_history}

# Current physician response (Type D: Health guidance)
{doctor_response}

Generate the physician's reasoning annotation covering:
1. Guideline source: evidence basis for the recommendation
2. Scientific rationale: underlying medical principles
3. Applicable scope: age range and symptom applicability
4. Specific instructions: actionable steps and key precautions

Output JSON: {{"inner_thinking": {{...}}}}
Only output the JSON object.""",

    "E": """You are a senior pediatrician at a tertiary children's hospital.

# Dialog history (up to current turn)
{dialog_history}

# Current physician response (Type E: Follow-up recommendation)
{doctor_response}

Generate the physician's reasoning annotation covering:
1. Follow-up purpose: why recheck is needed
2. Timing rationale: why this timeframe (drug kinetics, disease course)
3. Result interpretation: criteria for improvement, no change, or worsening
4. Subsequent actions: plan for each outcome scenario
5. Online follow-up: what can be handled remotely vs. needs in-person visit

Output JSON: {{"inner_thinking": {{...}}}}
Only output the JSON object.""",
}


def format_dialog_history(dialog_turns: list, up_to_index: int) -> str:
    """Format dialogue turns up to (not including) the current turn."""
    lines = []
    for i in range(up_to_index):
        turn = dialog_turns[i]
        role = "Patient/Parent" if turn["role"] == "patient" else "Physician"
        content = turn.get("content") or turn.get("content_zh", "")
        lines.append(f"[Turn {i}] {role}: {content}")
    return "\n".join(lines)

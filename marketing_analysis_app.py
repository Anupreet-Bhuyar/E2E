import streamlit as st
import json
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import re

# Page config
st.set_page_config(page_title="Marketing Component Analyzer", layout="wide")
st.title("🎯 Marketing Component Analyzer")

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Component definitions (40 marketing components)
COMP_DEFINITIONS = {
    "COMP1": "The ultimate transformation/end result the customer wants",
    "COMP2": "Social proof (testimonials, numbers, screenshots, credentials, case studies)",
    "COMP3": "Speed to first result/time-to-value",
    "COMP4": "Ease/simplification of the process",
    "COMP5": "Bonuses/extras that increase perceived value",
    "COMP6": "Specific pain points and frustrations",
    "COMP7": "Emotional and functional desires",
    "COMP8": "Objections and hesitations",
    "COMP9": "Real user phrases and language",
    "COMP10": "Target audience self-perception and traits",
    "COMP11": "Universal emotional truth behind behavior",
    "COMP12": "Root cause problem beneath surface complaints",
    "COMP13": "Core mechanism/why the product works",
    "COMP14": "Behavioral patterns and psychology",
    "COMP15": "Customer success story/anecdote",
    "COMP16": "Metrics, statistics, measurable outcomes",
    "COMP17": "Transparency about limitations and fit",
    "COMP18": "Ideal use cases and trigger moments",
    "COMP19": "Urgency/why now",
    "COMP20": "Attention-grabbing hook",
    "COMP21": "Old vs. new / myth vs. truth positioning",
    "COMP22": "Social proof visualization",
    "COMP23": "Specific credibility markers (names, numbers)",
    "COMP24": "Transformation journey model",
    "COMP25": "Storytelling narrative structure",
    "COMP26": "Guarantee or risk-reversal promise",
    "COMP27": "Point of view and story angle",
    "COMP28": "Core messages and key ideas",
    "COMP29": "Emotional undertone and feeling",
    "COMP30": "Brand voice and tone of voice",
    "COMP31": "Audience awareness level of problem",
    "COMP32": "Funnel stage and drop-off points",
    "COMP33": "Distribution channels",
    "COMP34": "Content format preferences",
    "COMP35": "Clear, simple statement rules",
    "COMP36": "Proof grounding for each message",
    "COMP37": "Real customer phrasing usage",
    "COMP38": "Emotional presence requirement",
    "COMP39": "Avoiding empty superlatives",
    "COMP40": "Implementation and execution guidelines"
}

# Funnel stages
FUNNEL_STAGES = {
    "TOF": "Top of Funnel (Awareness/Problem Recognition)",
    "MOF": "Middle of Funnel (Consideration/Solution)",
    "BOF": "Bottom of Funnel (Decision/Conversion)"
}

# Component-to-keywords mapping for detection
COMP_KEYWORDS = {
    "COMP1": ["transform", "achieve", "success", "result", "outcome", "become", "end-result"],
    "COMP2": ["testimonial", "review", "proof", "case study", "credential", "certification", "verified"],
    "COMP3": ["fast", "quick", "speed", "minutes", "hours", "instantly", "immediately"],
    "COMP4": ["easy", "simple", "effortless", "automated", "eliminates", "removes", "reduces effort"],
    "COMP6": ["frustrating", "struggle", "problem", "difficult", "painful", "frustrated", "overwhelm"],
    "COMP7": ["want", "desire", "crave", "aspire", "feel", "emotion", "emotional"],
    "COMP8": ["doubt", "hesitate", "skeptical", "worried", "concerned", "objection", "but"],
    "COMP9": ["slang", "metaphor", "phrase", "quote", "says", "told", "customer said"],
    "COMP13": ["because", "reason", "works", "mechanism", "secret", "unique", "special"],
    "COMP16": ["percent", "%", "number", "statistics", "data", "metrics", "increased"],
    "COMP19": ["now", "urgent", "today", "limited", "deadline", "hurry", "before"],
    "COMP20": ["hook", "attention", "grab", "stop", "catchy", "viral", "headline"],
}

def extract_components(input_text: str) -> dict:
    """Extract components from text using keyword matching and heuristics."""
    results = {}
    text_lower = input_text.lower()
    
    for comp_id, definition in COMP_DEFINITIONS.items():
        presence = False
        score = 0
        details = ""
        
        # Check for keywords
        if comp_id in COMP_KEYWORDS:
            keywords = COMP_KEYWORDS[comp_id]
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            if keyword_matches > 0:
                presence = True
                score = min(10, 3 + keyword_matches * 1.5)
                matches = [kw for kw in keywords if kw in text_lower]
                details = f"Found keywords: {', '.join(matches[:2])}"
        
        # Length-based quality heuristic
        if presence and len(input_text) < 50:
            score = max(0, score - 2)
        elif presence and len(input_text) > 500:
            score = min(10, score + 1)
        
        results[comp_id] = {
            "id": comp_id,
            "present": presence,
            "score": round(score, 1),
            "details": details,
            "definition": definition
        }
    
    return results

def analyze_funnel_stage(components: dict) -> dict:
    """Classify funnel stage based on component presence."""
    scores = {stage: 0 for stage in ["TOF", "MOF", "BOF"]}
    
    # Scoring logic
    tof_comps = ["COMP20", "COMP6", "COMP7", "COMP9"]
    mof_comps = ["COMP13", "COMP14", "COMP12", "COMP21"]
    bof_comps = ["COMP2", "COMP16", "COMP26", "COMP8"]
    
    for comp_id, data in components.items():
        if data["present"]:
            score_val = data["score"] / 10.0
            if comp_id in tof_comps:
                scores["TOF"] += score_val
            if comp_id in mof_comps:
                scores["MOF"] += score_val
            if comp_id in bof_comps:
                scores["BOF"] += score_val
    
    max_stage = max(scores, key=scores.get)
    confidence = scores[max_stage] / (sum(scores.values()) + 0.001)
    
    return {
        "funnel_stage": max_stage,
        "stage_description": FUNNEL_STAGES[max_stage],
        "scores": scores,
        "confidence": round(min(1.0, confidence), 2),
        "awareness_level": int(confidence * 5)
    }

def find_gaps(components: dict) -> list:
    """Identify missing or weak components."""
    gaps = []
    
    for comp_id, data in components.items():
        if not data["present"] or data["score"] < 5:
            gaps.append({
                "component": comp_id,
                "definition": data["definition"],
                "current_score": data["score"],
                "priority": "high" if not data["present"] else "medium"
            })
    
    # Sort by priority
    gaps.sort(key=lambda x: (x["priority"] == "medium", x["current_score"]))
    return gaps[:10]

def generate_recommendations(components: dict, gaps: list) -> dict:
    """Generate recommendations based on analysis."""
    
    weak_comps = [g["component"] for g in gaps if g["priority"] == "high"]
    
    recommendations = {
        "critical_gaps": weak_comps[:3],
        "quick_wins": [g["component"] for g in gaps[3:6] if g["priority"] == "medium"],
        "suggested_additions": [],
        "platform_suggestions": ["LinkedIn", "Email", "Website Hero", "Landing Page"],
        "next_steps": []
    }
    
    # Smart suggestions
    if "COMP2" in weak_comps:
        recommendations["next_steps"].append("Add customer testimonials or case studies")
    if "COMP3" in weak_comps:
        recommendations["next_steps"].append("Highlight speed/time-to-value")
    if "COMP16" in weak_comps:
        recommendations["next_steps"].append("Include metrics and statistics")
    if "COMP26" in weak_comps:
        recommendations["next_steps"].append("Add guarantee or risk-reversal statement")
    
    return recommendations

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Your Marketing Copy")
    input_text = st.text_area(
        "Paste your marketing copy, tagline, email, or ad copy:",
        height=200,
        placeholder="e.g., 'Transform your fitness in 30 days with our AI-powered coaching app...'"
    )

with col2:
    st.subheader("⚙️ Analysis Options")
    show_definitions = st.checkbox("Show component definitions", value=True)
    detailed_view = st.checkbox("Detailed view", value=False)

if st.button("🔍 Analyze Components", type="primary", use_container_width=True):
    if not input_text.strip():
        st.error("Please enter some marketing copy to analyze.")
    else:
        # Run analysis
        components = extract_components(input_text)
        funnel_data = analyze_funnel_stage(components)
        gaps = find_gaps(components)
        recommendations = generate_recommendations(components, gaps)
        
        st.session_state.analysis_results = {
            "components": components,
            "funnel": funnel_data,
            "gaps": gaps,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        st.success("Analysis complete!")

# Display results
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Funnel stage
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Funnel Stage", results["funnel"]["funnel_stage"])
    with col2:
        st.metric("Confidence", f"{results['funnel']['confidence']:.0%}")
    with col3:
        st.metric("Awareness Level", f"{results['funnel']['awareness_level']}/5")
    
    st.info(f"📍 {results['funnel']['stage_description']}")
    
    # Component breakdown
    st.divider()
    st.subheader("📊 Component Analysis")
    
    # Filter present components
    present = {k: v for k, v in results["components"].items() if v["present"]}
    
    if present:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{len(present)} components detected:**")
        
        for comp_id, data in sorted(present.items(), key=lambda x: x[1]["score"], reverse=True):
            cols = st.columns([1, 2, 1, 3])
            with cols[0]:
                st.write(f"**{comp_id}**")
            with cols[1]:
                st.progress(data["score"] / 10)
            with cols[2]:
                st.write(f"{data['score']}/10")
            with cols[3]:
                if show_definitions:
                    st.caption(data['definition'])
    
    # Gaps and recommendations
    st.divider()
    st.subheader("🎯 Gaps & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Critical Gaps:**")
        for gap in results["gaps"][:5]:
            st.write(f"- {gap['component']}: {gap['definition']}")
    
    with col2:
        st.write("**Recommended Actions:**")
        for action in results["recommendations"]["next_steps"]:
            st.write(f"✓ {action}")
    
    # Export results
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            "📥 Download JSON Results",
            json_str,
            "analysis_results.json",
            "application/json"
        )
    
    with col2:
        csv_data = "Component,Present,Score,Details\n"
        for comp_id, data in results["components"].items():
            csv_data += f"{comp_id},{data['present']},{data['score']},\"{data['details']}\"\n"
        st.download_button(
            "📥 Download CSV Results",
            csv_data,
            "analysis_results.csv",
            "text/csv"
        )

# Footer
st.divider()
st.caption("Marketing Component Analyzer v2.0 | Analyze and optimize your marketing copy")
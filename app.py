# app.py for Streamlit

import streamlit as st
import torch
import numpy as np
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from bert_score import score as bert_score
from sklearn.cluster import KMeans

# COMP descriptions (full dict for all 40)
comp_descriptions = {
    "COMP1": "What the customer ultimately wants (the final transformation). Questions: What is the perfect end-result your user dreams of achieving? How would users describe success in their own words? If everything worked ideally, what would their “after state” look like?",
    "COMP2": "Testimonials, numbers, screenshots, credentials, case studies. Questions: What existing proof do you have (numbers, screenshots, testimonials)? What internal metrics validate the product’s performance? What authority or certifications increase credibility?",
    "COMP3": "How fast the user gets their first meaningful result. Questions: How long before users typically experience their first win? What is the average time to full transformation? What steps reduce time-to-value?",
    "COMP4": "How easy you make things for the customer. Questions: What do users currently find hard? What steps does your product simplify or automate? How much time/effort is eliminated?",
    "COMP5": "Additional items that increase perceived value. Questions: What bonuses accelerate results? Which bonuses remove user friction? What “extras” do users gladly pay for separately?",
    "COMP6": "Frustrations, struggles, reasons things aren’t working. (Improved definition: Specific, emotionally rooted frustration that happens at a clear moment, causes anxiety/inconvenience, has real cost.) Questions: What issues do users complain about the most? What emotions (overwhelm, pressure, confusion) appear regularly? What does this pain cost the user?",
    "COMP7": "Emotional and functional wants. Questions: What emotional outcome do users crave the most? What functional outcome do they want faster/easier? What identity shift do they aspire to?",
    "COMP8": "Reasons they hesitate or don’t buy. Questions: What doubts keep users from purchasing? What misconceptions exist around the product? What objections appear in sales conversations?",
    "COMP9": "Real user phrases. Questions: What phrases appear repeatedly in reviews or chats? What slang or metaphors do users use? What exact words do people use to describe the problem?",
    "COMP10": "How they see themselves (MBTI, OCEAN, roles). Questions: How do users describe themselves? What group/tribe labels matter to them? What traits define their decision-making?",
    "COMP11": "Universal emotional truth behind buyer behavior. Questions: What universal emotion drives users in this category? What fears/desires shape behavior? What truth always applies?",
    "COMP12": "The real problem behind the surface-level complaint. Questions: What problem do users think they have? What deeper problem do they actually have? What do they consistently misdiagnose?",
    "COMP13": "The core reason the product works. Questions: What makes your solution truly work? What simple one-line mechanism explains your product? What’s the unique element others don’t have?",
    "COMP14": "Patterns behind how people behave. Questions: What repeated behaviors have you observed? What mistakes do users make before your product? What psychological pattern exists?",
    "COMP15": "A short anecdote that demonstrates value. Questions: Give one 1–2 sentence customer story. What moment best shows the product’s impact? What relatable scenario can communicate transformation?",
    "COMP16": "Stats, metrics, measurable outcomes. Questions: What numbers best show improvement? What measurable outcomes can you share? What percentage/time/cost changes matter most?",
    "COMP17": "Transparency builds trust. Questions: Who is NOT a good fit? When will the product fail? What expectations should you set?",
    "COMP18": "Where the product works best. Questions: When should users ideally use this? What scenario amplifies results? What “trigger moment” creates urgency?",
    "COMP19": "Why this matters right now. Questions: Why is “now” the best time to buy? What changed in the market? What happens if they wait?",
    "COMP20": "The hook that stops the scroll.",
    "COMP21": "Old way vs. new way, myth vs. truth.",
    "COMP22": "Showing others using or loving it.",
    "COMP23": "Names, numbers, details that build credibility.",
    "COMP24": "Transformation model.",
    "COMP25": "Classic storytelling flow.",
    "COMP26": "Guarantee or promise proving they won’t lose.",
    "COMP27": "Questions: What POV resonates best? What story angle matches brand identity? What angle is proven to convert?",
    "COMP28": "Questions: What 3–5 ideas MUST be communicated? What order best guides understanding? What message is mandatory?",
    "COMP29": "Questions: What emotion should messaging imply? What should the reader feel underneath the words? What psychological shift should happen?",
    "COMP30": "Questions: What tone aligns with your brand? What adjectives describe your brand? What tone builds the most trust?",
    "COMP31": "Questions: How aware is your target user of the problem? Do they know solutions exist? Do they know YOU?",
    "COMP32": "Questions: Where do prospects drop off most? What stage is most profitable? What stage lacks content currently?",
    "COMP33": "Questions: What channels do you use regularly? Which have the highest conversion? What channels does your audience prefer?",
    "COMP34": "Questions: What content formats work best? What formats do channels require? What formats do customers consume most?",
    "COMP35": "Clear, simple statements.",
    "COMP36": "Proof grounding each message.",
    "COMP37": "Use real customer phrasing.",
    "COMP38": "Emotion must be present.",
    "COMP39": "No empty superlatives.",
    "COMP40": "(Not explicitly defined in the provided text; appears to be part of the reference table but details are omitted.)"
}

# Microservices Prompts as Chains (using free gpt2 model)
llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation", model_kwargs={"max_length": 200})

# ComponentExtractorService Chain
extractor_prompt = PromptTemplate(
    template="""### Task: Analyze creative_text against COMP1–COMP40. Detect presence (binary), score quality (0-10: 0=irrelevant, 10=persuasive/aligned), extract details (1-2 sentences). Use brand_context for relevance.
### Context: Components are marketing elements. Definitions: {comp_defs}
### Multi-Step Reasoning:
1. Read inputs.
2. For each COMP: Scan text, match to definition/Qs using context.
3. Assign presence/score/details.
4. Anchor output as JSON: {{"components": [{{"id": "COMP1", "present": ..., "score": ..., "details": "..."}}, ...]}}.
Given input: {input_json}.""",
    input_variables=["comp_defs", "input_json"]
)
extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt)

# OfferAssemblerService Chain
assembler_prompt = PromptTemplate(
    template="""### Task: Assemble marketing offer from components via 6 steps. Produce core, layers, direction, context, 3-5 variants.
### Context: Offer = compelling proposition. Steps: 1. Map COMP1–19. 2. Core: Blend COMP1–5+13. 3. Persuasion: Add COMP20–26. 4. Direction: COMP27–30. 5. Mapping: COMP31–34. 6. Generate: Rules COMP35–40, variants.
### Multi-Step Reasoning:
1. Summarize inputs.
2. Execute steps sequentially.
3. Anchor output as JSON: {{"offer_core": "...", "persuasion_layers": [...], "creative_direction": {{...}}, "mapped_context": {{...}}, "generated_variants": ["...", ...]}}.
Given components: {input_json}.""",
    input_variables=["input_json"]
)
assembler_chain = LLMChain(llm=llm, prompt=assembler_prompt)

# FunnelClassifierService Chain
funnel_prompt = PromptTemplate(
    template="""### Task: Classify funnel (TOF/MOF/BOF) and awareness (1-5) with confidence (0-1). Provide reasons.
### Context: Awareness: 1=Unaware (curiosity). 2=Problem Aware (pain). 3=Solution (mechanism). 4=Product (proof). 5=Most Aware (offer). Funnel: TOF=curiosity/pain; MOF=mechanism/proof; BOF=offer/risk.
### Multi-Step Reasoning:
1. Weigh components (e.g., COMP20 high → TOF).
2. Calculate confidence (average matches).
3. Map to levels.
4. Anchor output as JSON: {{"funnel_stage": "...", "confidence": ..., "awareness_level": ..., "reasons": ["...", ...]}}.
Given creative: {input_json}.""",
    input_variables=["input_json"]
)
funnel_chain = LLMChain(llm=llm, prompt=funnel_prompt)

# DeliverableMapperService Chain
mapper_prompt = PromptTemplate(
    template="""### Task: Map offer to deliverable type, generate content sections using specified COMPs/rules.
### Context: Mappings: Homepage Hero: COMP1,2,3,6-7,9,13,16,27-30,35-36. [All mappings...]
### Multi-Step Reasoning:
1. Identify mapping for type.
2. Pull COMPs from input.
3. Generate sections step-by-step.
4. Anchor output as JSON: {{"deliverable_type": "...", "content_sections": {{"section1": "...", ...}}}}.
Given data/target: {input_json}.""",
    input_variables=["input_json"]
)
mapper_chain = LLMChain(llm=llm, prompt=mapper_prompt)

# PainPointAnalyzerService Chain
pain_prompt = PromptTemplate(
    template="""### Task: Extract pain points from data. Classify type, strength; link to COMP7/8.
### Context: Pain = specific/emotional/consequential/behavioral. Types: Functional/Emotional/Consequential. Formula: [User] feels [emotion] because [task] causing [consequence]. Strength: Frequent/emotional/measurable.
### Multi-Step Reasoning:
1. Scan data for frustrations.
2. Apply formula/types.
3. Evaluate strength.
4. Anchor output as JSON: {{"pain_points": [{{"description": "...", "type": "...", "strength": ...}}, ...], "linked_desires": [...], "linked_objections": [...]}}.
Given data: {input_json}.""",
    input_variables=["input_json"]
)
pain_chain = LLMChain(llm=llm, prompt=pain_prompt)

# RecommendationGeneratorService Chain
rec_prompt = PromptTemplate(
    template="""### Task: Generate gaps, platforms, usage recs, fixes from analysis.
### Context: Gaps = low/missing COMPs. Platforms from COMP33–34/funnel. Usage e.g., "Mid-funnel 0.68". Fixes = specific adds.
### Multi-Step Reasoning:
1. Identify weaknesses (low scores).
2. Suggest based on mappings/stages.
3. Anchor output as JSON: {{"gaps": ["..."], "platform_suggestions": ["..."], "usage_recommendations": "...", "fixes": ["..."]}}.
Given data: {input_json}.""",
    input_variables=["input_json"]
)
rec_chain = LLMChain(llm=llm, prompt=rec_prompt)

# TST Pipeline Tools

# Linguistic VAE (placeholder class, adapt from repo)
class LinguisticVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(384, 128)
        self.decoder = torch.nn.Linear(128, 384)

    def forward(self, input_emb):
        z = self.encoder(input_emb)
        return self.decoder(z), z

vae_model = LinguisticVAE()

def extract_disentangle(input_text, target_examples):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    source_emb = embedder.encode(input_text, convert_to_tensor=True)
    target_style = embedder.encode(target_examples, convert_to_tensor=True).mean(dim=0)
    z_content, z_style = vae_model(source_emb.unsqueeze(0))
    style_delta = target_style - z_style.squeeze()
    return z_content.squeeze().numpy(), style_delta.numpy()

extract_tool = Tool(
    name="ExtractDisentangle",
    func=extract_disentangle,
    description="Extract and disentangle style vectors using TextSETTR + Linguistic."
)

def mine_patterns(content_vector):
    kmeans = KMeans(n_clusters=3)
    patterns = kmeans.fit_predict(content_vector.reshape(1, -1))
    return [f"pattern_{p}" for p in patterns]

tokenizer = T5Tokenizer.from_pretrained('t5-base')
gen_model = T5ForConditionalGeneration.from_pretrained('t5-base')

def generate_candidates(input_text, style_delta, patterns):
    inputs = tokenizer(f"generate variant: {input_text} with delta: {style_delta.tolist()} patterns: {patterns}", return_tensors='pt')
    candidates = gen_model.generate(inputs.input_ids, penalty_alpha=0.6, top_k=4, num_return_sequences=10)
    return [tokenizer.decode(cand, skip_special_tokens=True) for cand in candidates]

generate_tool = Tool(
    name="MineGenerate",
    func=lambda input_text, style_delta: generate_candidates(input_text, style_delta, mine_patterns(style_delta)),
    description="Mine patterns with CTPM and generate diverse candidates with SimCTG."
)

tiny_styler = pipeline('text2text-generation', model='zacharyhorvitz/TinyStyler')

def refine_filter(candidates, comp_vectors):
    refined = []
    for cand in candidates:
        refined_text = tiny_styler(f"transfer to target: {cand}", max_length=100)[0]['generated_text']
        refs = list(comp_vectors.values())
        P, R, F1 = bert_score([refined_text] * len(refs), refs, lang="en")
        if F1.mean().item() > 0.70:
            refined.append(refined_text)
    return refined[:5]

refine_tool = Tool(
    name="RefineFilter",
    func=refine_filter,
    description="Refine with TinyStyler and filter for COMP consistency."
)

def evaluate_variants(variants, comp_descriptions):
    scores = {}
    for i, var in enumerate(variants):
        comp_scores = []
        for desc in comp_descriptions.values():
            P, R, F1 = bert_score([var], [desc], lang="en")
            comp_scores.append(F1.mean().item())
        scores[i+1] = (var, np.mean(comp_scores))
    return scores

evaluate_tool = Tool(
    name="EvaluateIntegrate",
    func=lambda variants: evaluate_variants(variants, comp_descriptions),
    description="Re-score for gaps and integrate into deliverables."
)

# Agent for TST Pipeline
tools = [extract_tool, generate_tool, refine_tool, evaluate_tool]
tst_agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Full Sequential Chain for End-to-End (Microservices + TST)
full_chain = SequentialChain(
    chains=[extractor_chain, assembler_chain, funnel_chain, mapper_chain, pain_chain, rec_chain],
    input_variables=["input_json"],
    output_variables=["extracted", "assembled", "funnel", "mapped", "pain", "recs"],
    verbose=True
)

# Run Function (inputs: ad_text, examples)
def run_pipeline(ad_text, examples):
    input_json = {"creative_text": ad_text, "brand_context": "Marketing AI tool"}
    comp_defs = "\n".join([f"- {k}: {v}" for k, v in comp_descriptions.items()])
    extracted = extractor_chain.run({"comp_defs": comp_defs, "input_json": input_json})
    assembled = assembler_chain.run({"input_json": extracted})
    funnel = funnel_chain.run({"input_json": assembled})
    mapped = mapper_chain.run({"input_json": funnel})
    pain = pain_chain.run({"input_json": mapped})
    recs = rec_chain.run({"input_json": pain})
    
    # TST Part
    tst_input = {"input": ad_text, "target_examples": examples}
    tst_result = tst_agent.run(tst_input)
    
    report = f"""
Extracted Components: {extracted}
Assembled Offer: {assembled}
Funnel Classification: {funnel}
Mapped Deliverables: {mapped}
Pain Points: {pain}
Recommendations: {recs}
TST Variants: {tst_result}
"""
    return report

# Streamlit UI
st.title("Ad Variant Generator")

ad_text = st.text_area("Enter Ad Copy", value="Discover the power of AI in marketing. Create ads that convert in minutes!")
examples_text = st.text_area("Enter Few-Shot Examples (one per line)", value="Unlock your marketing potential with AI tools.\nTired of low conversion? AI is here to help.")
uploaded_file = st.file_uploader("Upload Ad Copy Text File (optional)", type="txt")

if uploaded_file is not None:
    ad_text = uploaded_file.read().decode("utf-8")

if st.button("Generate Variants & Report"):
    examples = examples_text.split("\n")
    with st.spinner("Processing..."):
        result = run_pipeline(ad_text, examples)
    st.subheader("Report and Variants")
    st.text(result)

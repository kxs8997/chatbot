import streamlit as st
import random
from datasets import load_dataset
import evaluate
from PIL import Image

# Import your project modules.
from image_processing import generate_caption_for_image, model as moondream_model
from qa_chain import CustomOllama
from utils import extract_thoughts

# For cosine similarity.
from sentence_transformers import SentenceTransformer, util
# For fuzzy matching.
from rapidfuzz import fuzz

# --------------------------
# Caching functions
# --------------------------

@st.cache_data(show_spinner=False)
def load_docvqa_dataset(split="validation", limit=4000):
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    if len(dataset) > limit:
        dataset = dataset.select(range(limit))
    return dataset

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def init_deepseek(base_url="http://localhost:11434", temperature=0.6, max_tokens=8192):
    return CustomOllama(
        model="deepseek-r1:70b",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )

@st.cache_resource(show_spinner=False)
def load_meteor():
    return evaluate.load("meteor")

# --------------------------
# Evaluation helper functions
# --------------------------

def compute_cosine_similarity(prediction: str, reference: str, embedder) -> float:
    emb_pred = embedder.encode(prediction, convert_to_tensor=True)
    emb_ref = embedder.encode(reference, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(emb_pred, emb_ref)
    return cos_sim.item()

def fuzzy_exact_match_presence(prediction: str, reference: str, threshold: int = 80) -> (bool, float):
    ratio = fuzz.partial_ratio(reference.lower(), prediction.lower())
    return ratio >= threshold, ratio

def evaluate_sample(sample, llm, embedder, meteor_metric, caption_threshold=20):
    # Extract sample data.
    image = sample["image"]
    question = sample["question"]
    ground_truth = sample["answers"][0] if isinstance(sample["answers"], list) else sample["answers"]

    # Generate a detailed caption.
    caption = generate_caption_for_image(image, caption_length="detailed", debug=False)
    #st.write("**Initial Detailed Caption:**", caption)
    
    # If the caption is too short, directly ask Moondream to rewrite it.
    if len(caption.split()) < caption_threshold:
        #st.write(f"Caption is shorter than {caption_threshold} words. Refining caption...")
        requery_prompt = (
            f"Rewrite the image caption with more detail to fully answer the question: '{question}'. "
            f"Current caption: '{caption}'."
        )
        #st.write("**Direct Requery Prompt for Moondream:**", requery_prompt)
        refined_result = moondream_model.query(image, requery_prompt)
        refined_caption = refined_result.get("answer", "")
        #st.write("**Refined Caption from Moondream:**", refined_caption)
        caption = refined_caption

    # Build final prompt for DeepSeek.
    final_prompt = f"Document Caption: {caption}\nQuestion: {question}\nAnswer:"
    #st.write("**Final Prompt for DeepSeek:**", final_prompt)

    # Get the final answer from DeepSeek.
    raw_answer = llm(final_prompt)
    _, main_answer = extract_thoughts(raw_answer)

    # Compute evaluation metrics.
    cos_sim = compute_cosine_similarity(main_answer, ground_truth, embedder)
    match_present, match_score = fuzzy_exact_match_presence(main_answer, ground_truth)
    meteor_result = meteor_metric.compute(predictions=[main_answer], references=[ground_truth])
    meteor_score = meteor_result.get("meteor", 0)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "caption": caption,
        "final_prompt": final_prompt,
        "predicted_answer": main_answer,
        "cosine_similarity": cos_sim,
        "fuzzy_exact_match": "Yes" if match_present else "No",
        "fuzzy_match_score": match_score,
        "meteor_score": meteor_score,
        "image": image
    }

# --------------------------
# Streamlit App Interface
# --------------------------

st.title("Mabl-GPT Vision - Text Evaluation Demo")
st.write("Press the button below to evaluate a random sample from the DocVQA dataset.")

# Load dataset and models.
dataset = load_docvqa_dataset(limit=100)
embedder = get_embedder()
llm = init_deepseek()
meteor_metric = load_meteor()

if st.button("Evaluate Random Sample"):
    sample = random.choice(dataset)
    result = evaluate_sample(sample, llm, embedder, meteor_metric, caption_threshold=20)
    
    # Display the sample image first.
    st.image(result["image"], caption="Sample Image", use_container_width=True)
    
    st.subheader("Evaluation Results")
    st.markdown(f"**Question:** {result['question']}")
    st.markdown(f"**Ground Truth:** {result['ground_truth']}")
    st.markdown(f"**Caption Used:** {result['caption']}")
    st.markdown(f"**Final Prompt Sent to DeepSeek:** {result['final_prompt']}")
    st.markdown(f"**Predicted Answer:** {result['predicted_answer']}")
    st.markdown(f"**Cosine Similarity:** {result['cosine_similarity']:.4f}")
    st.markdown(f"**Fuzzy Exact Match:** {result['fuzzy_exact_match']} (Score: {result['fuzzy_match_score']:.2f})")
    st.markdown(f"**METEOR Score:** {result['meteor_score']:.4f}")

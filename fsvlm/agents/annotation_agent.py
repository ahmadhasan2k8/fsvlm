"""Annotation Agent — classifies user defect descriptions into training data.

Takes an AnnotationSession (user's clicked-and-described defect regions) and:
1. Uses the base VLM to classify descriptions into a defect taxonomy
2. Generates standardized training descriptions referencing location + type
3. Outputs LabeledSamples ready for the training pipeline

This agent bridges the gap between "user points at problems" and
"structured training data the Training Agent can consume."
"""

from __future__ import annotations

from loguru import logger

from fsvlm.config import FSVLMConfig
from fsvlm.types import (
    AnnotationSession,
    LabeledSample,
)

# Prompt for the LLM to classify user descriptions into defect types
_CLASSIFY_PROMPT = """\
You are a quality control expert. Below are defect descriptions from a user \
annotating images of the same type of product. Group them into defect categories.

User descriptions:
{descriptions}

Respond with a JSON object mapping each defect type name to a one-sentence \
definition. Use short, lowercase names (e.g. "crack", "scratch", "contamination"). \
Merge descriptions that refer to the same type of defect. Output ONLY the JSON, \
no markdown fences.

Example output:
{{"crack": "A visible crack or fracture in the surface", \
"stain": "Discoloration or foreign material on the surface"}}
"""

# Prompt to classify a single description into one of the taxonomy types
_ASSIGN_PROMPT = """\
Given these defect categories:
{taxonomy}

Which category best fits this description: "{description}"

Respond with ONLY the category name, nothing else.
"""


class AnnotationAgent:
    """Processes user annotations into structured training data.

    Args:
        config: FSVLM configuration.
    """

    def __init__(self, config: FSVLMConfig) -> None:
        self.config = config

    def classify_annotations(
        self,
        session: AnnotationSession,
    ) -> AnnotationSession:
        """Classify all annotations into a defect taxonomy using the base VLM.

        Mutates the session in-place: fills defect_taxonomy and each
        annotation's defect_type field.

        Args:
            session: AnnotationSession with user descriptions.

        Returns:
            The same session with taxonomy and defect_types populated.
        """
        # Collect all unique descriptions
        descriptions = []
        for img in session.images:
            for ann in img.annotations:
                if ann.user_description.strip():
                    descriptions.append(ann.user_description.strip())

        if not descriptions:
            logger.warning("No descriptions to classify")
            return session

        # Deduplicate for the classification prompt
        unique_descs = list(dict.fromkeys(descriptions))
        logger.info("Classifying {} unique descriptions", len(unique_descs))

        # Build taxonomy using LLM — always use the VLM so it understands
        # the user's descriptions the way they intend them
        taxonomy = self._build_taxonomy(unique_descs)
        session.defect_taxonomy = taxonomy
        logger.info("Defect taxonomy: {}", taxonomy)

        # Assign each annotation to a type
        type_names = list(taxonomy.keys())
        for img in session.images:
            for ann in img.annotations:
                if ann.user_description.strip() and type_names:
                    ann.defect_type = self._assign_type(ann.user_description, taxonomy)

        return session

    def annotations_to_samples(
        self,
        session: AnnotationSession,
    ) -> list[LabeledSample]:
        """Convert classified annotations to LabeledSamples for training.

        Each annotation becomes its OWN training sample — same image, different
        description. This means one image with 3 defect clicks = 3 defect
        samples, each teaching the model to recognize a different issue
        (or different location of the same issue).

        Good images (is_good=True) become "good" samples.

        Args:
            session: Classified AnnotationSession.

        Returns:
            List of LabeledSamples ready for training.
        """
        samples: list[LabeledSample] = []

        for img in session.images:
            if img.is_good:
                samples.append(
                    LabeledSample(
                        image_path=img.image_path,
                        label="good",
                        description="The item appears to be in good condition with no visible defects.",
                    )
                )
                continue

            if not img.annotations:
                continue

            # Each annotation is its own training sample
            for ann in img.annotations:
                defect_type = ann.defect_type or "defect"
                location = ann.location_description or "the image"

                if ann.user_description:
                    description = (
                        f"FAIL\n{defect_type.capitalize()} detected in {location}: {ann.user_description}"
                    )
                else:
                    description = f"FAIL\n{defect_type.capitalize()} detected in {location}."

                samples.append(
                    LabeledSample(
                        image_path=img.image_path,
                        label="defect",
                        description=description,
                    )
                )

        logger.info(
            "Generated {} samples ({} defect, {} good)",
            len(samples),
            sum(1 for s in samples if s.label == "defect"),
            sum(1 for s in samples if s.label == "good"),
        )
        return samples

    def _build_taxonomy(self, descriptions: list[str]) -> dict[str, str]:
        """Use base VLM to group descriptions into defect categories."""
        import json as json_mod

        desc_text = "\n".join(f"- {d}" for d in descriptions)
        prompt = _CLASSIFY_PROMPT.format(descriptions=desc_text)

        response = self._query_llm(prompt)

        # Parse JSON from response
        try:
            # Strip any markdown fences if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            taxonomy = json_mod.loads(cleaned)
            if isinstance(taxonomy, dict):
                return {str(k): str(v) for k, v in taxonomy.items()}
        except (json_mod.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse taxonomy JSON: {}. Using fallback.", e)

        # Fallback: create one type per unique description keyword
        return self._fallback_taxonomy(descriptions)

    def _assign_type(
        self,
        description: str,
        taxonomy: dict[str, str],
    ) -> str:
        """Assign a single description to the best matching defect type."""
        type_names = list(taxonomy.keys())

        # If only one type, use it
        if len(type_names) == 1:
            return type_names[0]

        # Simple keyword matching first (fast, no LLM call)
        desc_lower = description.lower()
        for name in type_names:
            if name.lower() in desc_lower:
                return name

        # Fall back to LLM classification
        taxonomy_text = "\n".join(f"- {name}: {defn}" for name, defn in taxonomy.items())
        prompt = _ASSIGN_PROMPT.format(taxonomy=taxonomy_text, description=description)

        response = self._query_llm(prompt).strip().lower()

        # Find best match
        for name in type_names:
            if name.lower() in response:
                return name

        # Default to first type
        return type_names[0]

    def _query_llm(self, prompt: str) -> str:
        """Query the base VLM as a text-only LLM.

        Uses FastVisionModel since the default model is a vision model.
        For text-only prompts, we just skip the image input.
        """
        import gc

        import torch

        model_name = self.config.default_model
        logger.debug("Loading base model for classification: {}", model_name)

        try:
            from unsloth import FastVisionModel

            model, tokenizer = FastVisionModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            FastVisionModel.for_inference(model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        # Text-only chat (no image)
        chat = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        device = next(model.parameters()).device
        inputs = tokenizer(text=prompt_text, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        return response

    @staticmethod
    def _fallback_taxonomy(descriptions: list[str]) -> dict[str, str]:
        """Create a simple taxonomy from keywords when LLM parsing fails."""
        # Extract likely defect type words
        common_types = [
            "crack",
            "scratch",
            "dent",
            "hole",
            "stain",
            "chip",
            "break",
            "contamination",
            "discoloration",
            "deformation",
            "rust",
            "corrosion",
            "tear",
            "missing",
            "bent",
            "burn",
        ]

        found: dict[str, str] = {}
        for desc in descriptions:
            desc_lower = desc.lower()
            for t in common_types:
                if t in desc_lower and t not in found:
                    found[t] = f"{t.capitalize()} defect"
                    break

        if not found:
            found["defect"] = "General defect"

        return found

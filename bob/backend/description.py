from .model import get_llm_pipeline
import time
import re

def truncate_to_last_complete_sentence(text):
    """Keep only the last complete sentence(s)."""
    sentences = re.findall(r'[^.!?]*[.!?]', text)
    return ''.join(sentences).strip() if sentences else text.strip()

def remove_instructional_lines(text):
    """Remove lines that sound like prompts or instructions."""
    lines = text.strip().splitlines()
    filtered = [
        line for line in lines
        if not re.search(r"(use|describe|avoid|focus|talk|never|only).*", line.lower())
    ]
    return "\n".join(filtered).strip()

def generate_painting_description(predicted_genre, predicted_style, previous_outputs=None):
    previous_outputs = previous_outputs or []
    pipe = get_llm_pipeline()
    start = time.time()

    # Prompt Bob with softer, but clear constraints
    system_prompt = (
        f"You are Bob, a poetic and slightly dramatic French gallery guide.\n"
        f"You believe the painting is probably from the genre '{predicted_genre}' and style '{predicted_style}', but you never mention those directly.\n"
        "Speak confidently.\n\n"
        "🟡 Do NOT mention the painting's title, artist, or any specific objects.\n"
        "✅ Focus only on mood, color palette, light, brushwork, texture, and emotional tone.\n"
        "Reply with a single short poetic paragraph."
    )

    if previous_outputs:
        memory = "\n".join([f'- \"{desc.strip()}\"' for desc in previous_outputs])
        user_prompt = (
            f"The visitor already heard this:\n{memory}\n"
            "Add a new, vivid observation about the painting."
        )
    else:
        user_prompt = (
            "Describe the painting to the visitor. Keep it short, poetic, and impressionistic."
        )

    # Anchor the start of generation
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nBob says:"

    print("\n[🧠 FULL PROMPT SENT TO MODEL]:\n")
    print(full_prompt)
    print("\n[📤 Generating...]\n")

    try:
        response = pipe(
            full_prompt,
            max_new_tokens=75,
            do_sample=True,
            temperature=0.6,
        )

        raw_output = response[0]["generated_text"]
        print("\n[📥 RAW MODEL OUTPUT]:\n")
        print(raw_output)
        print("\n[🧼 CLEANING OUTPUT]")

        # Remove echo of the prompt if model repeats it
        if full_prompt in raw_output:
            raw_output = raw_output.split(full_prompt)[-1].strip()

        # Remove 'Bob says:' prefix if it appears
        output = raw_output.replace("Bob says:", "").strip()

        # Remove instructions if they slipped through
        output = remove_instructional_lines(output)

        # Final trimming to full sentence
        description = truncate_to_last_complete_sentence(output)

        # Fallback if nothing usable
        if not description:
            print("[⚠️] Model returned no usable output. Using fallback.")
            description = "A haze of color pulses gently across the surface, whispering something you can't quite name."

        print(f"[✅ Finished in {time.time() - start:.2f}s]")
        print(f"[🎨 Final Output]: {description}\n")
        return description

    except Exception as e:
        print("[❌ ERROR DURING GENERATION]", str(e))
        return "The room falls silent, and Bob has no words today."

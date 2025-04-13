import os
import io
import time
import json
import random
import base64
import torch
from flask import current_app, render_template, request, redirect, url_for, session
import torchvision.transforms as transforms
from backend.description import generate_painting_description  # assuming description.py remains in backend/

# Label mappings
genre_names = [
    "Abstract Painting", "Cityscape", "Genre Painting", "Illustration", 
    "Landscape", "Nude Painting", "Portrait", "Religious Painting", 
    "Sketch And Study", "Still Life", "Unknown Genre"
]

style_names = [
    "Abstract Expressionism", "Art Nouveau", "Baroque", "Expressionism", 
    "Impressionism", "Naive Art Primitivism", "Northern Renaissance", 
    "Post Impressionism", "Realism", "Romanticism", "Symbolism"
]

def init_routes(app):
    
    @app.route('/')
    def main():
        session.pop("sampled_indices", None)
        return render_template('main.html')

    @app.route('/selection')
    def selection():
        if "sampled_indices" not in session:
            indices = random.sample(current_app.valid_indices, 5)
            session["sampled_indices"] = indices
            current_app.image_cache.clear()
            current_app.description_cache.clear()
            current_app.user_guesses.clear()
            print("🔄 Sampled new images:", indices)
        else:
            indices = session["sampled_indices"]
            print("➡️ Reusing cached indices:", indices)

        samples = []
        for idx in indices:
            if idx in current_app.image_cache:
                img_str = current_app.image_cache[idx]
            else:
                buffer = io.BytesIO()
                current_app.dataset[idx]["image"].save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                current_app.image_cache[idx] = img_str

            sample = current_app.dataset[idx]
            samples.append({
                "img_str": img_str,
                "index": idx,
                "genre": sample.get("genre", "unknown"),
                "style": sample.get("style", "unknown"),
                "has_guess": idx in current_app.user_guesses
            })

        all_guessed = all(idx in current_app.user_guesses for idx in indices)
        return render_template("selection.html", samples=samples, all_guessed=all_guessed)


    @app.route('/challenge')
    def challenge():
        start_time = time.time()
        index = request.args.get("index")
        if not index:
            return redirect(url_for('selection'))
        index = int(index)

        sample = current_app.dataset[index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(sample["image"]).unsqueeze(0).to(device)

        with torch.no_grad():
            genre_logits, style_logits = current_app.model(img_tensor)
        pred_genre = torch.argmax(genre_logits, dim=1).item()
        pred_style = torch.argmax(style_logits, dim=1).item()

        genre_name = genre_names[pred_genre] if pred_genre < len(genre_names) else 'unknown'
        style_name = style_names[pred_style] if pred_style < len(style_names) else 'unknown'

        # Initialize description history for this index if needed
        if index not in current_app.description_cache:
            current_app.description_cache[index] = []

        generated_description = generate_painting_description(
            genre_name, style_name, previous_outputs=current_app.description_cache[index]
        )
        current_app.description_cache[index].append(generated_description)

        if index in current_app.image_cache:
            img_str = current_app.image_cache[index]
        else:
            buffer = io.BytesIO()
            sample["image"].save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            current_app.image_cache[index] = img_str

        def format_label(idx, names):
            try:
                return f"{idx} - {names[idx]}"
            except (IndexError, TypeError):
                return f"{idx} - unknown"

        gt_genre = sample.get("genre", "unknown")
        gt_style = sample.get("style", "unknown")

        return render_template('challenge.html',
            index=index,
            img_str=img_str,
            generated_description=generated_description,
            predicted_genre=f"{pred_genre} - {genre_name}",
            predicted_style=f"{pred_style} - {style_name}",
            gt_genre=format_label(gt_genre, genre_names),
            gt_style=format_label(gt_style, style_names),
            history=current_app.description_cache[index]
        )

    @app.route('/submit_guess', methods=["POST"])
    def submit_guess():
        index = int(request.form["index"])
        guess = request.form["guess"]
        current_app.user_guesses[index] = guess
        print(f"[USER GUESS] Index {index}: {guess}")
        return redirect(url_for("selection"))

    @app.route('/score')
    def score():
        total, correct = 0, 0
        detailed = []

        for index, guess in current_app.user_guesses.items():
            sample = current_app.dataset[index]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(sample["image"]).unsqueeze(0).to(device)

            with torch.no_grad():
                genre_logits, style_logits = current_app.model(img_tensor)
            pred_genre = torch.argmax(genre_logits, dim=1).item()
            pred_style = torch.argmax(style_logits, dim=1).item()

            gt_genre = sample.get("genre")
            gt_style = sample.get("style")

            if guess == "OK":
                correct_flag = (pred_genre == gt_genre) and (pred_style == gt_style)
            elif guess == "NOT OK":
                correct_flag = (pred_genre != gt_genre) and (pred_style != gt_style)
            elif guess == "GENRE":
                correct_flag = (pred_genre != gt_genre) and (pred_style == gt_style)
            elif guess == "STYLE":
                correct_flag = (pred_genre == gt_genre) and (pred_style != gt_style)
            else:
                correct_flag = False

            total += 1
            correct += int(correct_flag)

            # Use image cache (same logic as in selection)
            if index in current_app.image_cache:
                img_str = current_app.image_cache[index]
            else:
                buffer = io.BytesIO()
                sample["image"].save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                current_app.image_cache[index] = img_str

            detailed.append({
                "index": index,
                "guess": guess,
                "correct": correct_flag,
                "pred_genre": genre_names[pred_genre],
                "pred_style": style_names[pred_style],
                "gt_genre": genre_names[gt_genre],
                "gt_style": style_names[gt_style],
                "img_str": img_str  # Include the cached image string here
            })

        return render_template("score.html", score=correct, total=total, detailed=detailed)

    @app.route('/detail')
    def detail():
        # Get image index from query
        index = request.args.get("index")
        if index is None:
            return redirect(url_for('score'))
        index = int(index)
        
        sample = current_app.dataset[index]
        
        # Process the image (using same transformation as in challenge/score)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(sample["image"]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Run the main model predictions
            genre_logits, style_logits = current_app.model(img_tensor)
        pred_genre_idx = torch.argmax(genre_logits, dim=1).item()
        pred_style_idx = torch.argmax(style_logits, dim=1).item()
        
        # Retrieve ground truth from the sample
        gt_genre = sample.get("genre")
        gt_style = sample.get("style")
        
        # Determine correctness using the user's guess (if available)
        guess = current_app.user_guesses.get(index, "OK")
        if guess == "OK":
            correct_flag = (pred_genre_idx == gt_genre) and (pred_style_idx == gt_style)
        elif guess == "NOT OK":
            correct_flag = (pred_genre_idx != gt_genre) and (pred_style_idx != gt_style)
        elif guess == "GENRE":
            correct_flag = (pred_genre_idx != gt_genre) and (pred_style_idx == gt_style)
        elif guess == "STYLE":
            correct_flag = (pred_genre_idx == gt_genre) and (pred_style_idx != gt_style)
        else:
            correct_flag = False

        # Obtain cached image
        if index in current_app.image_cache:
            img_str = current_app.image_cache[index]
        else:
            buffer = io.BytesIO()
            sample["image"].save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            current_app.image_cache[index] = img_str

        # Call the second model to get additional descriptive text.
        try:
            from backend.description import generate_additional_text
            additional_text = generate_additional_text(sample["image"])
        except ImportError:
            additional_text = ""

        # Prepare prediction and ground truth strings using label mappings
        pred_genre = genre_names[pred_genre_idx]
        pred_style = style_names[pred_style_idx]
        gt_genre_str = genre_names[gt_genre] if gt_genre < len(genre_names) else "unknown"
        gt_style_str = style_names[gt_style] if gt_style < len(style_names) else "unknown"

        return render_template("detail.html",
                            index=index,
                            img_str=img_str,
                            pred_genre=pred_genre,
                            pred_style=pred_style,
                            gt_genre=gt_genre_str,
                            gt_style=gt_style_str,
                            guess=guess,
                            additional_text=additional_text,
                            correct=correct_flag)

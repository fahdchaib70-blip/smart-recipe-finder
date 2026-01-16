import os
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000/search")
TOP_K = 3

# ---------------------------------------------------------------------
# UI Header
# ---------------------------------------------------------------------
st.title("Smart Recipe Finder 🥦")
st.write(
    """
Découvrez de nouvelles recettes adaptées à vos envies et à vos ingrédients !
Que vous cherchiez de l’inspiration ou une idée rapide, cette application alimentée par l’IA
vous propose des recommandations pertinentes à partir de votre requête.
"""
)
st.divider()

# ---------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------
user_input = st.text_input(
    "Qu’avez-vous envie de cuisiner aujourd’hui ? Décrivez votre idée et on vous aide !"
)


def _remove_video_links_section(text: str) -> str:
    """Remove the 'Video Links:' block if it exists in the generated response."""
    if not text:
        return ""
    marker = "Video Links:"
    return text.split(marker)[0].strip() if marker in text else text.strip()


def _call_backend(query: str, top_k: int = TOP_K) -> dict:
    """Send the search request to the backend and return parsed JSON."""
    response = requests.post(
        BACKEND_URL,
        json={"query": query, "top_k": top_k},
        timeout=30,  # prevents the UI from hanging forever
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------
# Action button
# ---------------------------------------------------------------------
if st.button("Obtenir des recommandations !"):
    if not user_input.strip():
        st.warning("Veuillez entrer une requête pour obtenir des recommandations.")
        st.stop()

    with st.spinner("Recherche de délicieuses idées..."):
        try:
            data = _call_backend(user_input.strip(), TOP_K)

            query = data.get("query", "")
            response_text = data.get("response", "")
            recipes = data.get("recipes", []) or []
            videos = data.get("videos", {}) or {}

            # No results
            if not recipes:
                st.error("Aucune recette pertinente n'a été trouvée. Essayez une autre requête.")
                st.stop()

            # -----------------------------------------------------------------
            # AI Suggestion
            # -----------------------------------------------------------------
            st.markdown("### **Suggestion de l'IA :**")

            cleaned_response = _remove_video_links_section(response_text)
            if cleaned_response:
                html_content = f"""
                <div style="text-align: justify; margin-bottom: 20px; line-height: 1.6;">
                    {cleaned_response}
                </div>
                """
                st.markdown(html_content, unsafe_allow_html=True)
            else:
                st.markdown("Aucune réponse disponible.")

            # -----------------------------------------------------------------
            # Recommended recipes
            # -----------------------------------------------------------------
            st.markdown("### **Recettes recommandées :**")

            for recipe in recipes:
                title = (recipe.get("title") or "Recette sans nom").capitalize()
                ingredients = recipe.get("ingredients", "")
                directions = recipe.get("directions", "Aucune direction disponible")
                recipe_id = recipe.get("id", "")

                # If ingredients arrives as a list, display as a single string
                if isinstance(ingredients, list):
                    ingredients = ", ".join(ingredients)

                # Recipe card
                recipe_html = f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px;
                            margin-bottom: 15px; background-color: #f9f9f9;">
                    <h4 style="color: #074E0A;">{title}</h4>
                    <p><b>Ingrédients :</b> {ingredients if ingredients else 'Aucun ingrédient disponible.'}</p>
                </div>
                """
                st.markdown(recipe_html, unsafe_allow_html=True)

                # Expandable details: directions + video
                with st.expander(f"Directions pour {title}"):
                    if directions and directions != "Aucune direction disponible":
                        steps = [s.strip() for s in directions.split(".") if s.strip()]
                        for i, step in enumerate(steps, start=1):
                            st.markdown(f"{i}. {step}")
                    else:
                        st.write("Aucune direction disponible.")

                    if recipe_id and recipe_id in videos:
                        st.markdown("### **Tutoriel vidéo**")
                        st.markdown(f"[Regarder le tutoriel ici]({videos[recipe_id]})", unsafe_allow_html=True)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération des recettes : {e}")
        except ValueError as e:
            st.error(f"Erreur d'analyse de la réponse du backend : {e}")
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite : {e}")

# app.py

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

class AppClassification:
    """Application Streamlit pour classifier des images"""

    def __init__(self, model_path, class_indices):
        # Charger le modèle Keras
        self.model = load_model(model_path)
        # Inverser le dictionnaire class_indices pour avoir {index -> nom de classe}
        self.classes = {v: k for k, v in class_indices.items()}

    def predire(self, img):
        """Prédit la classe d'une image PIL"""
        img = img.convert('RGB')               # S'assurer que l'image est en RGB
        img = img.resize((224, 224))          # Redimensionner à la taille du modèle
        img_array = np.array(img) / 255.0     # Normaliser les pixels
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension batch

        predictions = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = self.classes[class_idx]

        return class_name, confidence, predictions[0]

    def run(self):
        """Lance l'application Streamlit"""
        
        # CSS personnalisé pour un design moderne
        st.markdown("""
        <style>
        /* Amélioration générale */
        .main {
            padding: 2rem;
        }
        
        /* Style pour le titre principal */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        /* Container pour le titre */
        .title-container {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Bouton de téléchargement amélioré */
        .stFileUploader > div > div {
            border-radius: 15px;
            border: 2px dashed #667eea;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div > div:hover {
            border-color: #764ba2;
            background-color: #f8f9ff;
        }
        
        /* Images avec ombre */
        .stImage img {
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease;
        }
        
        .stImage img:hover {
            transform: scale(1.02);
        }
        
        /* Boxes de résultats */
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
            margin: 1rem 0;
        }
        
        .confidence-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 8px 16px rgba(245, 87, 108, 0.3);
            margin: 1rem 0;
        }
        
        /* Amélioration des sous-titres */
        h3 {
            color: #667eea;
            margin-top: 2rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        /* Animation spinner */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Métriques améliorées */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Titre avec design moderne
        st.markdown("""
        <div class="title-container">
            <h1>🌸 Classificateur de Fleurs 🌸</h1>
            <p style="font-size: 1.2rem; color: #666;">Powered by Deep Learning & MobileNetV2</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section principale
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            uploaded_file = st.file_uploader(
                "📎 Choisissez une image de fleur",
                type=['jpg', 'jpeg', 'png'],
                help="Formats acceptés: JPG, JPEG, PNG"
            )

        if uploaded_file is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Affichage de l'image uploadée
                img = Image.open(uploaded_file)
                st.image(img, caption='📷 Image uploadée', use_column_width=True)

            st.markdown("---")
            
            # Prédiction avec spinner personnalisé
            with st.spinner('🔍 Analyse de l\'image en cours...'):
                class_name, confidence, all_probs = self.predire(img)
            
            # Affichage des résultats avec design amélioré
            st.markdown("""
            <style>
            .stSuccess {
                border-radius: 15px;
                padding: 1.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Métriques stylisées
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🌺 Classe Prédite",
                    value=class_name.upper()
                )
            
            with col2:
                st.metric(
                    label="📊 Confiance",
                    value=f"{confidence*100:.2f}%"
                )
            
            st.markdown("---")
            
            # Probabilités pour toutes les classes
            st.markdown("### 📈 Probabilités Détaillées par Classe")
            
            prob_df = pd.DataFrame({
                'Classe': [self.classes[i] for i in range(len(all_probs))],
                'Probabilité (%)': all_probs * 100
            }).sort_values('Probabilité (%)', ascending=False)
            
            # Graphique en barres avec couleurs personnalisées
            st.bar_chart(prob_df.set_index('Classe'), use_container_width=True)
            
            # Tableau amélioré
            st.markdown("#### 📋 Détails des Probabilités")
            styled_df = prob_df.style.background_gradient(
                cmap='RdYlGn', 
                subset=['Probabilité (%)']
            ).format({'Probabilité (%)': '{:.2f}%'})
            st.dataframe(styled_df, use_container_width=True)
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div class="footer">
                <p>Classificateur de fleurs développé avec Streamlit & TensorFlow</p>
            </div>
            """, unsafe_allow_html=True)


# -------------------
# Lancement de l'application
# -------------------
if __name__ == '__main__':
    # Définir class_indices pour les fleurs
    class_indices = {'daisy': 0, 'rose': 1}

    app = AppClassification('mon_classificateur.h5', class_indices)
    app.run()

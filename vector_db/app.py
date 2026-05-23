import sys

import numpy as np
import plotly.express as px
import torch
from PIL import Image
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from qdrant_client import QdrantClient
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoProcessor

from consts import CLIP_FOLDER, QDRANT_COLLECTION_NAME, QDRANT_LOCATION


class VectorMapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLIP Creep")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.info_label = QLabel("Load an image to search the database")
        layout.addWidget(self.info_label)

        self.upload_btn = QPushButton("Upload Image & Find Matches")
        self.upload_btn.clicked.connect(self.upload_and_search)
        layout.addWidget(self.upload_btn)

        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

        self.processor = None
        self.model = None
        self.qdrant = None

    def init_ai_models(self):
        if self.model is None:
            self.info_label.setText("Loading AI models")
            QApplication.processEvents()
            self.processor = AutoProcessor.from_pretrained(CLIP_FOLDER)
            self.model = AutoModel.from_pretrained(CLIP_FOLDER)
            self.qdrant = QdrantClient(QDRANT_LOCATION)

    def upload_and_search(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select a Dog Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        self.upload_btn.setEnabled(False)
        self.init_ai_models()

        try:
            self.info_label.setText("Analyzing image...")
            QApplication.processEvents()

            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            query_vector = outputs.pooler_output.tolist()[0]

            self.info_label.setText("Searching database...")
            QApplication.processEvents()

            hits = self.qdrant.query_points(
                collection_name=QDRANT_COLLECTION_NAME, query=query_vector, limit=15
            )

            top_breeds = []
            for hit in hits.points:
                breed = hit.payload.get("breed", "Unknown").split("-")[-1]
                if breed not in top_breeds:
                    top_breeds.append(breed)
                if len(top_breeds) == 3:
                    break

            self.info_label.setText(
                f"Top breeds: {', '.join(top_breeds)}. Fetching map data..."
            )
            QApplication.processEvents()

            records, _ = self.qdrant.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                limit=5000,
                with_vectors=True,
                with_payload=True,
            )

            vectors = []
            labels = []
            sizes = []

            for r in records:
                vectors.append(r.vector)
                b_name = r.payload.get("breed", "Unknown").split("-")[-1]

                if b_name == top_breeds[0]:
                    labels.append(f"1st Match: {b_name}")
                elif len(top_breeds) > 1 and b_name == top_breeds[1]:
                    labels.append(f"2nd Match: {b_name}")
                elif len(top_breeds) > 2 and b_name == top_breeds[2]:
                    labels.append(f"3rd Match: {b_name}")
                else:
                    labels.append("Other Breeds")

                sizes.append(3)

            vectors.append(query_vector)
            labels.append("YOUR UPLOADED IMAGE")
            sizes.append(15)

            self.info_label.setText("Running t-SNE to build 3D map...")
            QApplication.processEvents()

            vectors_np = np.array(vectors)
            tsne = TSNE(n_components=3, perplexity=50, random_state=42, max_iter=1000)
            vectors_3d = tsne.fit_transform(vectors_np)

            self.info_label.setText("Rendering 3D Plot...")
            QApplication.processEvents()

            color_map = {
                "YOUR UPLOADED IMAGE": "red",
                f"1st Match: {top_breeds[0]}": "green",
                f"2nd Match: {top_breeds[1]}"
                if len(top_breeds) > 1
                else "None1": "blue",
                f"3rd Match: {top_breeds[2]}"
                if len(top_breeds) > 2
                else "None2": "yellow",
                "Other Breeds": "lightgrey",
            }

            fig = px.scatter_3d(
                x=vectors_3d[:, 0],
                y=vectors_3d[:, 1],
                z=vectors_3d[:, 2],
                color=labels,
                color_discrete_map=color_map,
                size=sizes,
                size_max=15,
                hover_name=labels,
                title="Semantic Search Results",
                opacity=0.8,
            )

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"
                ),
                legend_title="Legend",
            )

            raw_html = fig.to_html(include_plotlyjs="cdn")
            self.web_view.setHtml(raw_html)

            self.info_label.setText(f"Success! Top Matches: {', '.join(top_breeds)}")

        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")
            print(f"Error details: {e}")

        finally:
            self.upload_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VectorMapApp()
    window.show()
    sys.exit(app.exec())

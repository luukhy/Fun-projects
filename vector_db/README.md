Worth reading
- PCA / t-SNE
- metalearning
- proto Net - "loss functinos separating clusters apart"
- FSL - few shot learning (DO PJMatch warto)
- Learning from few examples: a summary of approaches to few-shot learning


Suggenstions:
- CLIP model
- sam (segment anything (?), google)


task: 
model clip (or other), pass a dataset (like Stanford Dogs) through the model, extract embeddings and place them into a vector database

- sam pipeline 3 pkt
- z gui - 5 pkt

# Vector Database Creep

## Idea
The application lets you creep inside a vision model, see what the model converts the input into (a mutlidimentional vector) and look at the vector in a simplified 3D-representation of the whole vector space. This lets you see what cluster will be asociated with the input.

## UI
The UI lets you choose any JPG image from your computer and displays the output vector in a simplified 3D representation of the vector space of the model after the inference.

## Tech Stack
- UI: PySide6, matplotlib (plotting)
- Vector Database: Qdrant

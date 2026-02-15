# Warehouse Robot Intelligent Perception System
---
## Overview

This project implements a prototype AI system for warehouse robotics that combines:

- Computer vision for object detection and tracking  
- Machine learning for object classification  
- Retrieval-based knowledge system for handling instructions  
- End-to-end integration into a unified workflow  

The system demonstrates how warehouse robots can perceive objects, classify them, and retrieve operational guidance dynamically.

---

## Project Structure
```
part1_cv/ → Vision detection & tracking
part2_ml/ → Classification model training & inference
part3_rag/ → Knowledge base retrieval system
part4_integration/ → Final integrated demo
data/ → Dataset used for training/testing
models/ → Saved trained models
results/ → Screenshots, metrics, demo outputs
```

---

## Setup Instructions

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dependencies

Main libraries used:

- OpenCV for computer vision  
- TensorFlow/Keras for classification model  
- NumPy for numerical processing  
- LangChain & sentence-transformers for retrieval system  
- FAISS for vector search  

All dependencies are listed in `requirements.txt`.

---

## How to Run Each Component

### Part 1 – Vision Module

```python part1_cv/object_detection.py```


Detects and tracks objects in webcam feed.

---

### Part 2 – Train Classifier

```python part2_ml/train_model.py```


Trains classification model and saves it to `/models`.

---

### Part 2 – Test Classifier

```python part2_ml/classify_image.py```


Runs inference on sample image.

---

### Part 3 – Knowledge Retrieval System


```python part3_rag/rag_system.py```

Allows querying warehouse instructions from documentation.

---

### Part 4 – Full Integrated System

```python part4_integration/integration_demo.py```


Runs full pipeline:

```vision → classification → instruction retrieval```

---

## Results

### Part 1 – Object Detection
![Part 1](results/screenshots/part1_detection.png)

### Part 2 – Model Training & Metrics
![Part 2](results/screenshots/part2_training_metrics.png)

### Part 3 – Knowledge Retrieval (RAG)
![Part 3](results/screenshots/part3_rag.png)

### Part 4 – Integrated System Demo
![Part 4](results/screenshots/part4_integration.png)


---

## Challenges Faced & Solutions

**1. Dataset availability**  
Real warehouse images were unavailable.  
**Solution:** Generated a synthetic dataset to validate the ML pipeline.

**2. Integration delays**  
Retrieval system caused latency during demo.  
**Solution:** Cached instructions locally to ensure responsive interaction.

**3. Model accuracy limitations**  
Small dataset led to approximate predictions.  
**Solution:** Focused on validating the architecture rather than optimizing accuracy.

**4. Dependency compatibility**  
Some libraries changed import paths during development.  
**Solution:** Updated imports and environment setup accordingly.

---

## Results

The system successfully demonstrates:

- Real-time object detection and tracking  
- End-to-end ML inference pipeline  
- Retrieval of handling instructions based on predicted category  
- Integrated AI workflow suitable for warehouse robotics  

Sample outputs, metrics, and demo recordings are available in the `results/` folder.

---

## Author

Arman Singh  
AI Research Internship Submission – Warehouse Robotics Prototype



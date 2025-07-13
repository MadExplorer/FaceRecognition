# Face Recognition Pipeline

This project implements a complete face recognition system in five stages:

1. **Face Detection**

   - **Method:** OpenCV Haar Cascade (frontal faces)

2. **Keypoint Localization**

   - **Model:** Stacked Hourglass Network (2 stacks, residual connections)
   - **Output:** Gaussian heatmaps for each landmark

3. **Face Alignment**

   - **Approach:** Affine transform based on three landmarks (left eye, right eye, nose)
   - **Result:** Cropped faces standardized in size and orientation

4. **Embedding Extraction**

   - **Backbone:** ResNet-18 (ImageNet pretrained, final layer replaced)
   - **Head:** ArcFace margin-based classifier
   - **Output:** L2-normalized feature vectors

5. **Evaluation & Demo**

   - **Procedure:** Compare embeddings using cosine distance
   - **Expectation:** Lower distances for same identities, higher for different

---

## Directory Structure

```
/ FaceRecognition
├─ code/                   # Python scripts and modules
├─ pictures/               # Visual outputs
│  ├─ detections/          # Detected faces
│  ├─ landmarked_faces/    # Faces with keypoints drawn
│  └─ processed_faces/     # Aligned and standardized faces
├─ README.md               # This document
```

## Requirements

- Python 3.8 or higher
- PyTorch
- torchvision
- OpenCV (cv2)
- numpy, pandas

Install dependencies via:

```
pip install torch torchvision opencv-python numpy pandas
```

## Usage

1. Place model files in project root:
   - `landmark_model.pth`
   - `arcface_model.pth`
2. Run the pipeline script:

```
python code/pipeline.py
```

---

Feel free to replace any stage (detection, keypoint model, embedder) with alternative methods as needed.

---

## Publishing on GitHub

To make the repository more complete and user-friendly, consider adding:

1. **License**: Choose an open-source license (e.g., MIT, Apache 2.0) and include a `LICENSE` file.
2. **Requirements File**: Add `requirements.txt` listing exact package versions for reproducibility.
3. **.gitignore**: Include common ignores (e.g., `__pycache__/`, `*.pyc`, model weight files if large).
4. **Badges**: Display build status, Python version, license, coverage badges at the top of `README.md`.
5. **Usage Examples**: Add code snippets and screenshots under a `## Examples` section.
6. **Contribution Guidelines**: Create `CONTRIBUTING.md` with instructions for issue reporting and pull requests.
7. **CI/CD**: Configure GitHub Actions workflow (e.g., run linting, tests) in `.github/workflows/`.
8. **Docker Support**: Provide a `Dockerfile` for a containerized environment.
9. **Documentation**: Add detailed docstrings or a `docs/` folder if expanding.
10. **Changelog**: Maintain `CHANGELOG.md` to track feature additions and fixes.

With these additions, your project will be ready for collaboration and easier onboarding on GitHub!


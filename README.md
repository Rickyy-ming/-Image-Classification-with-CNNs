🧠 Advanced Image Classification with CNNs (CIFAR-10 & CIFAR-100)

📌 Project Overview

This project develops an advanced Convolutional Neural Network (CNN) architecture using PyTorch to classify images in the CIFAR-10 and CIFAR-100 datasets. The core focus is to enhance model generalization and interpretability using state-of-the-art deep learning strategies including:
	•	Residual connections (ResNet-inspired)
	•	Dropout and Batch Normalization
	•	Adaptive learning rate scheduling
	•	Data augmentation
	•	Grad-CAM for explainability
🎯 Objectives
	•	Build a deep CNN architecture incorporating modern best practices
	•	Evaluate model performance on both CIFAR-10 and CIFAR-100
	•	Compare model performance with existing baselines in literature
	•	Use Grad-CAM to visually interpret how the CNN learns from images

⸻

💡 Value Proposition
	•	Demonstrates proficiency in custom CNN design and training optimization
	•	Bridges theoretical deep learning knowledge with real-world model deployment
	•	Adds an interpretable and reproducible framework for vision tasks, ideal for ML and data science roles
	•	Applicable to a wide range of real-world use cases like medical imaging, autonomous vehicles, or smart manufacturing

⸻

🧪 Techniques Used

📦 Dataset
	•	CIFAR-10: 10 classes of 32x32 color images (e.g., airplane, car, dog, truck)
	•	CIFAR-100: Similar structure but with 100 fine-grained object classes
	•	Datasets loaded using torchvision.datasets and preprocessed with PyTorch transforms

🔄 Data Preprocessing & Augmentation
	•	RandomCrop with padding
	•	RandomHorizontalFlip
	•	ColorJitter for realistic brightness/contrast variations
	•	Pixel normalization with transforms.Normalize

🧠 CNN Architecture Highlights
	•	Multiple convolution blocks with:
	•	Residual connections
	•	Batch normalization
	•	Dropout layers
	•	ReLU activations and MaxPooling
	•	Fully connected layers with final Softmax classification
	•	Modular design for easy experimentation with deeper/lighter models

⚙️ Training Techniques
	•	Optimizer: Adam
	•	Loss Function: CrossEntropyLoss
	•	Scheduler: ReduceLROnPlateau (dynamic adjustment based on validation loss)
	•	Device: torch.device("cuda") for GPU acceleration

📊 Evaluation Metrics
	•	Accuracy
	•	Precision / Recall / F1 Score (macro and per-class)
	•	Confusion matrix visualizations using Seaborn

🧠 Model Explainability
	•	Implemented Grad-CAM to generate heatmaps showing which parts of the image influenced the model’s decision
	•	Visualization helps interpret misclassifications and assess bias/robustness

⸻

🔮 Future Enhancements
	•	Integrate pretrained architectures (ResNet-50, EfficientNet) via transfer learning
	•	Experiment with CIFAR-100 for more fine-grained classification
	•	Hyperparameter tuning using Optuna or Ray Tune
	•	Streamlit or Flask app deployment with real-time prediction upload interface


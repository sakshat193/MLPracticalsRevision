# ML–II Practical Exam Preparation (Exam-Ready Notebooks)

This repository is a **single workspace** of exam-ready notebooks for ML–II practicals.

Design goals:
- **Minimal and explicit**: avoid hidden magic; show weights/bias, loops, metrics.
- **Exam-output aligned**: each notebook prints/plots what examiners usually ask for.
- **Reusable**: during an exam you should mostly change dataset loading + a few hyperparameters.

## Quick Start (Windows / PowerShell)

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install requirements
```powershell
pip install -r requirements.txt
```

### 3) Open and run notebooks
Open any `.ipynb` in VS Code and select the `.venv` kernel.

### 4) Sanity-check the environment (optional)
```powershell
python -c "import numpy, pandas, sklearn, tensorflow, keras, matplotlib, seaborn; import minisom; print('OK')"
```

## Notebook Index (By Question)

### Topic 1 — Perceptron
- **Q1.1**: `BooleanLogicGatesImplementation.ipynb` — AND/OR/NOT/NAND/NOR + XOR explanation/solution
- **Q1.2**: `MultiOutputPerceptron.ipynb` — AND + OR simultaneously using a weight matrix
- **Q1.3**: `PerceptronConvergenceComparison.ipynb` — convergence comparison (epochs / behavior)

### Topic 2 — Autoencoders
- **Q2.1**: `IrisAutoencoderRepresentation.ipynb` — 2D latent space + reconstruction visualization
- **Q2.2**: `IrisAutoencoderNeuronComparison.ipynb` — 2-neuron vs 3-neuron bottleneck + MSE
- **Q2.3**: `HeartDiseaseAutoencoder.ipynb` — simple vs deep autoencoder comparison
- **Q2.4**: `WineDatasetAutoencoder.ipynb` — 3-neuron bottleneck + loss curve
- **Q2.5**: `TitanicAutoencoderSparseVsDenoising.ipynb` — sparse (L1) vs denoising (GaussianNoise)
- **Q2.6**: `IrisFeatureNormalization.ipynb` — Min–Max scaling + before/after plots

### Topic 3 — CNN
- **Q3.1**: `MNISTCNNOptimizerComparison.ipynb` — Adam vs Adagrad + confusion matrix/metrics
- **Q3.2**: `FashionMNISTHyperparameterTuning.ipynb` — tuning (LR/batch/epochs) + optimizer comparison
- **Q3.3**: `CIFAR10DataAugmentation.ipynb` — with/without augmentation + overfitting discussion
- **Q3.4**: `IntelImageClassificationCNN.ipynb` — folder-based dataset + confusion matrix
- **Q3.5**: `AlpacaClassificationCNN.ipynb` — strict Conv → Pool → Flatten → Dense + curves
- **Q3.6**: `CornDiseaseClassificationCNN.ipynb` — explicit preprocessing + multi-class evaluation
- **Q3.7**: `CIFAR100OptimizerComparison.ipynb` — optimizer comparison + accuracy/loss plots
- **Q3.8**: `MNIST5OptimizerComparison.ipynb` — compare 5 optimizers (same epochs/batch)
- **Q3.9**: `SimCLRFramework.ipynb` — SimCLR + NT-Xent + representation visualization

### Topic 4 — Transfer Learning
- **Q4.1**: `TransferLearningComparison.ipynb` — frozen vs fine-tuned (VGG16/ResNet50)

### Topic 5 — Gradient Descent / Optimizers
- **Q5.1**: `CaliforniaHousingErrorSurface.ipynb` — manual GD + 3D error surface
- **Q5.2**: `GradientDescentVariants.ipynb` — SGD vs Batch vs Mini-batch vs Momentum

### Topic 6 — Mathematical Fundamentals
- **Q6.1**: `BackpropagationFromScratch.ipynb` — backprop on Iris from scratch
- **Q6.2**: `ManualBackpropCalculation.ipynb` — step-by-step gradients + MAE/MSE + loss surface
- **Q6.3**: `ActivationFunctionsPlot.ipynb` — activations + derivatives + MAE/MSE

### Topic 7 — Self-Organizing Maps
- **Q7.1**: `SOMCreditCardFraud.ipynb` — SOM tuning + U-matrix + anomaly marking

## Examiner Expectations (Checklist)

Use this section as a quick “did I print/plot everything?” checklist.

### Common outputs to always include
- **Model parameters**: final weights/bias (perceptron, GD, backprop).
- **Metrics**: accuracy for classification; MSE/MAE for regression where applicable.
- **Plots**:
	- Loss vs epochs / iterations
	- Training vs validation accuracy/loss for CNNs
	- Confusion matrix for image classification questions

### Key question-specific notes

**Q1.1 XOR**
- Must state: XOR is **not linearly separable** → single-layer perceptron cannot solve it.

**Q2 autoencoders**
- Normalize inputs, show bottleneck values, show reconstruction error (MSE), plot loss.

**Q3.5 Alpaca**
- Architecture must contain exactly the required layer types (Conv, Pool, Flatten, Dense).

**Q3.6 Corn**
- Explicit resizing step (e.g., 224×224) and normalization are crucial.

**Q3.7 CIFAR-100**
- Accuracy will typically be lower than CIFAR-10; focus on the comparison between optimizers.

**Q3.9 SimCLR**
- Must show two augmented views and implement NT-Xent contrastive loss.

## Reusable Minimal Templates (Copy-Paste)

These blocks are intentionally short and exam-friendly.

### Perceptron (logic gates)
```python
import numpy as np

def step(z):
		return 1 if z >= 0 else 0

def train_p(X, y, lr=0.1, epochs=20):
		w = np.zeros(X.shape[1])
		b = 0.0
		for _ in range(epochs):
				for i in range(len(X)):
						y_hat = step(float(X[i] @ w + b))
						w += lr * (y[i] - y_hat) * X[i]
						b += lr * (y[i] - y_hat)
		return w, b
```

### Autoencoder (dense)
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler

X = MinMaxScaler().fit_transform(X)
inp = Input(shape=(X.shape[1],))
z = Dense(3, activation='relu')(inp)
out = Dense(X.shape[1], activation='sigmoid')(z)
ae = Model(inp, out)
ae.compile(optimizer='adam', loss='mse')
ae.fit(X, X, epochs=50, verbose=0)
```

### Batch Gradient Descent (from scratch)
```python
import numpy as np

def batch_gd(X, y, lr=0.01, epochs=50):
		w = np.zeros(X.shape[1])
		b = 0.0
		loss_hist = []
		m = len(X)
		for _ in range(epochs):
				y_hat = X @ w + b
				err = y_hat - y
				w -= lr * (X.T @ err) / m
				b -= lr * err.mean()
				loss_hist.append((err ** 2).mean())
		return w, b, loss_hist
```

### ROC curve (binary)
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
```

### SOM
```python
from minisom import MiniSom

som = MiniSom(10, 10, X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)
```

## Dataset Setup (Folder-based Image Datasets)

Some notebooks require you to download datasets locally and update paths.

### Recommended local layout
Create a folder like:
```
datasets/
	intel_images/
	alpaca/
	corn_dataset/
```

### Expected folder structures

Intel Images (Q3.4):
```
seg_train/
	buildings/
	forest/
	glacier/
	mountain/
	sea/
	street/
seg_test/
	buildings/
	forest/
	glacier/
	mountain/
	sea/
	street/
```

Alpaca (Q3.5):
```
alpaca/
	alpaca/
	not_alpaca/
```

Corn 3-classes (Q3.6):
```
corn_dataset/
	Common_Rust/
	Gray_Leaf_Spot/
	Healthy/
```

### What to edit in notebooks
- Look for `data_dir`, `train_dir`, and `test_dir` variables near the top.
- Point them to your local dataset folders.

## Practical Run Tips

- First run: set small epochs (e.g., 2–3) to confirm everything works.
- Then increase epochs for nicer curves.
- For confusion matrices: ensure evaluation generators use `shuffle=False`.
- If your machine is slow, reduce dataset size (especially for CIFAR-100 and SimCLR).

## Troubleshooting

### Kernel / `.venv` not showing in VS Code
- Ensure VS Code extensions: Python + Jupyter.
- Re-open VS Code after creating `.venv`.

### TensorFlow install problems
- This repo targets Python 3.10.
- If TF fails to install, update `pip`:
	```powershell
	python -m pip install --upgrade pip
	```

### Dataset not found
- Update the notebook path variables.
- Keep datasets under `datasets/` (ignored by git) to avoid committing large files.

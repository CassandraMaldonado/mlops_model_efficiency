# Model Efficiency

This repository contains experiments focused on improving model efficiency for deployment by using Knowledge Distillation and Quantization techniques.
The work was completed as part of the Model Performance and Optimization module in the MS-ADS MLOps course.

**1. Knowledge Distillation**


⸻

**2. Quantization**

1) Which quantization method did not run? Why?

Static Post-Training Quantization (Static-PTQ) failed.
NotImplementedError occurred: quantized::conv2d.new unsupported on CPU backend.
Cause: missing quantized backend kernels (e.g., FBGEMM or QNNPACK) or Colab’s PyTorch build lacking support.

2) What changes were needed for Dynamic PTQ and QAT?
	•	Force CPU-only execution; disable mixed precision.
	•	Fuse modules (Conv+BN+ReLU).
	•	For Dynamic PTQ, quantize only nn.Linear layers:

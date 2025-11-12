# Model Efficiency

**1. Knowledge Distillation**

1) Why do you need soft-probabilities (the teacher’s output distribution)?

Hard labels (one-hot targets) only say which class is correct. By contrast, the teacher’s soft probabilities carry “dark knowledge”—they encode inter-class similarity.
Example: for a husky image, a good teacher might assign non-zero probability to wolf and fox.
Those relative probabilities provide richer gradients for the student, improving sample efficiency and generalization.
In practice:
	•	They stabilize training (each class contributes a small gradient).
	•	They guide the student toward the teacher’s decision boundaries rather than only the final answer.

2) Why was the Student loss = NaN, and how to correct it?

In the uploaded notebook, the student loss became NaN due to:
	•	Mismatched scales (not dividing by temperature).
	•	Log of zeros or unnormalized tensors in KL.
	•	Mixed precision overflow when T was large.

Fixes:
	•	Apply temperature on both sides:
p_t = softmax(teacher_logits / T)
log_p_s = log_softmax(student_logits / T)
	•	Multiply KD loss by T**2.
	•	Keep KD in float32, disable AMP.
	•	Use gradient clipping (clip_grad_norm_).

3) What is the purpose of the temperature T?

T controls the smoothness of the teacher’s distribution:
	•	High T (e.g., 10.0) → softer, reveals inter-class similarity.
	•	Low T (e.g., 0.7) → sharper, more like one-hot labels.

Multiplying by T² stabilizes gradient magnitudes.

4) What happens when T = 10.0 vs T = 0.7?
	•	T=10.0: very smooth; gradients shrink without scaling, slower learning.
	•	T=0.7: sharp, faster initial learning but loses “dark knowledge.”

Notebook results:
Teacher error improved 34% → 28%,
Student latency ~2.8 ms vs 7.5 ms (teacher).
Models saved: resnet18_fp32.pth, resnet18_kd.pth, KD+QAT variants.

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

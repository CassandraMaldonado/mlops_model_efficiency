# Model Efficiency

This repository contains experiments focused on improving model efficiency for deployment by using Knowledge Distillation and Quantization techniques.

## Overview

The project explores how to make deep learning models smaller and faster without losing much accuracy.

Two main approaches were implemented and tested:

**1. Knowledge Distillation (KD):** training a smaller student model using the output distribution from a larger trained teacher model.
	
**2. Quantization:** converting floating-point models into lower-precision representations to reduce memory and inference latency.

The experiments were conducted on CIFAR-10 using ResNet architectures, implemented in PyTorch.

**1. Knowledge Distillation**

The goal was to transfer knowledge from a ResNet-50 teacher to a ResNet-18 student.

	•	The teacher model was first trained on CIFAR-10 to convergence, achieving improved accuracy over baseline runs.
	•	The student model was then trained using both hard labels and soft probabilities from the teacher model’s logits.
	•	Distillation training was tested with different temperature values and loss weightings to balance the teacher and student objectives.
	•	The initial run produced unstable loss values (NaN) which were later fixed by normalizing the temperature scaling and stabilizing the KL divergence computation.
	•	Once stable, the student model achieved nearly the same accuracy as the teacher while being substantially smaller and faster.
	•	Latency measurements showed a reduction from approximately 7.5 ms (teacher) to 2.8 ms (student).
	•	Final models were saved for both teacher and student, including distilled and hybrid variants:
	•	resnet18_fp32.pth
	•	resnet18_kd.pth
	•	resnet18_kd_qat.pth

⸻

**2. Quantization**


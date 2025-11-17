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

•	Final models were saved for both teacher and student, including distilled and hybrid variants: resnet18_fp32.pth, resnet18_kd.pth and resnet18_kd_qat.pth.


**2. Quantization**

This section focused on evaluating Post-Training Quantization (PTQ), Dynamic Quantization and Quantization-Aware Training (QAT) to further compress and accelerate models.

•	The Static PTQ attempt failed due to PyTorch backend limitations in the runtime environment, specifically the missing CPU kernel for quantized::conv2d.new.

•	Dynamic Quantization and QAT were implemented successfully after adjusting the evaluation flow to use CPU-only inference.

	•	A CPU evaluation function was added since quantized models cannot run on GPU.
	•	The qnnpack and fbgemm backends were tested for compatibility.
	•	Module fusion (Conv + BN + ReLU) was applied to prepare the model for quantization.
	•	Dynamic Quantization was applied primarily to linear layers, achieving improved efficiency without the static kernel issues.
	•	Quantization-Aware Training (QAT) was performed for two epochs to finetune fake-quantized weights before conversion.
	•	Models were benchmarked for accuracy, latency, and size:
	•	Quantized models showed significant model size reduction (from ~42 MB to ~10 MB) with minimal accuracy drop.
	•	Latency improved further under the quantized configurations.


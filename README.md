# Multimodal Large Language Models Safety

This repository provides a brief summary of algorithms from our review paper [Survey of Adversarial Robustness in Multimodal Large Language Models](arXiv preprint arXiv:2503.13962). 

Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

See our paper for more details.



# Attack on Image-MLLMs 

## Adversarial Attack

#### Full Access Situation

- On the adversarial robustness of multi-modal foundation models. \[[paper](https://openaccess.thecvf.com/content/ICCV2023W/AROW/html/Schlarmann_On_the_Adversarial_Robustness_of_Multi-Modal_Foundation_Models_ICCVW_2023_paper.html)\]
- On the robustness of large multimodal models against image adversarial attacks. \[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Cui_On_the_Robustness_of_Large_Multimodal_Models_Against_Image_Adversarial_CVPR_2024_paper.html)\]
- Image Hijacks: Adversarial Images can Control Generative Models at Runtime. \[[paper](https://proceedings.mlr.press/v235/bailey24a.html)\]
- Inducing High Energy-Latency of Large Vision-Language Models with Verbose Images. \[[paper](https://openreview.net/forum?id=BteuUysuXX)\]
- Stop reasoning! when multimodal LLMs with chain-of-thought reasoning meets adversarial images. \[[paper](https://openreview.net/forum?id=oqYiYG8PtY#discussion)\]

#### Restricted Access Situation

- On evaluating adversarial robustness of large vision-language models. \[[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a97b58c4f7551053b0512f92244b0810-Abstract-Conference.html)\]
- Efficient Generation of Targeted and Transferable Adversarial Examples for Vision-Language Models Via Diffusion Models. \[[paper](https://ieeexplore.ieee.org/abstract/document/10812818)\]
- How Many Are in This Image A Safety Evaluation Benchmark for Vision LLMs. \[[paper](https://link.springer.com/chapter/10.1007/978-3-031-72983-6_3)\]
- How Robust is Google's Bard to Adversarial Image Attacks? \[[paper](https://openreview.net/forum?id=qtpTVc1c3c)\]
- Dissecting Adversarial Robustness of Multimodal LM Agents. \[[paper](https://openreview.net/forum?id=YauQYh2k1g)\]
- B-AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Black-box Adversarial Visual-Instructions. \[[paper](https://ieeexplore.ieee.org/abstract/document/10816024)\]

### Jailbreak Attacks

#### Cross-Modal Inconsistency Attacks

- Are aligned neural networks adversarially aligned? \[[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c1f0b856a35986348ab3414177266f75-Abstract-Conference.html)\]
- Jailbreaking attack against multimodal large language model. \[[paper](https://openreview.net/forum?id=pucOtwqaLB)\]
- Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast. \[[paper](https://dl.acm.org/doi/abs/10.5555/3692070.3692731)\]
- ImgTrojan: Jailbreaking Vision-Language Models with ONE Image. \[[paper](https://arxiv.org/abs/2403.02910)\]

#### Prompt Manipulation

- White-box multimodal jailbreaks against large vision-language models. \[[paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681092)\]
- Jailbreaking GPT-4v via self-adversarial attacks with system prompts. \[[paper](https://arxiv.org/abs/2311.09127)\]
- Figstep: Jailbreaking large vision-language models via typographic visual prompts. \[[paper](https://arxiv.org/abs/2311.05608)\]

#### Jailbreak in Training-Phase

- Images are achilles’ heel of alignment: Exploiting visual vulnerabilities for jailbreaking multimodal large language models. \[[paper](https://link.springer.com/chapter/10.1007/978-3-031-73464-9_11)\]
- Visual adversarial examples jailbreak aligned large language models. \[[paper](https://ojs.aaai.org/index.php/AAAI/article/view/30150)\]

### Data Integrity Attacks

- Test-Time Backdoor Attacks on Multimodal Large Language Models. \[[paper](https://arxiv.org/abs/2402.08577)\]
- VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models. \[[paper](https://link.springer.com/article/10.1007/s11263-025-02368-9)\]
- Physical Backdoor Attack can Jeopardize Driving with Vision-Large-Language Models. \[[paper](https://openreview.net/forum?id=gPmKbViJ6o)\]
- Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models. \[[paper](https://openreview.net/forum?id=JhqyeppMiD)\]
- Revisiting Backdoor Attacks against Large Vision-Language Models. \[[paper](https://openreview.net/forum?id=muzK5zp4Vq)\]



# Attack on Video-MLLMs

### Temporal-Coherence Attack

- Fmm-attack: A flow-based multi-modal adversarial attack on video-based LLMs. \[[paper](https://arxiv.org/abs/2403.13507)\]
- Beyond Raw Videos: Understanding Edited Videos with Large Multimodal Model. \[[paper](https://arxiv.org/abs/2406.10484)\]

### Scenario-Driven Attacks

- Pg-attack: A precision-guided adversarial attack framework against vision foundation models for autonomous driving. \[[paper](https://arxiv.org/abs/2407.13111)\]
- Visual Adversarial Attack on Vision-Language Models for Autonomous Driving. \[[paper](https://arxiv.org/abs/2411.18275)\]
- Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography. \[[paper](https://arxiv.org/abs/2405.14169)\]

### Benchmarks and Evaluation

- T2VSafetyBench: Evaluating the Safety of Text-to-Video Generative Models. \[[paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/74eed5f568354c2e77dd9b018f38a9d4-Abstract-Datasets_and_Benchmarks_Track.html)\]
- Beyond Raw Videos: Understanding Edited Videos with Large Multimodal Model. \[[paper](https://arxiv.org/abs/2406.10484)\]
- AV-Deepfake1M: A large-scale LLM-driven audio-visual deepfake dataset. \[[paper](https://dl.acm.org/doi/abs/10.1145/3664647.3680795)\]
- VideoQA in the Era of LLMs: An Empirical Study. \[[paper](https://link.springer.com/article/10.1007/s11263-025-02385-8)\]



# Attack on Audio-MLLMs

- BadRobot: Manipulating Embodied LLMs in the Physical World. \[[paper](https://arxiv.org/abs/2407.20242)\]
- (Ab) using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs. \[[paper](https://arxiv.org/abs/2307.10490)\]
- Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Language Models. \[[paper](https://arxiv.org/abs/2411.14842)\]
- TrustNavGPT: Modeling Uncertainty to Improve Trustworthiness of Audio-Guided LLM-Based Robot Navigation. \[[paper](https://ieeexplore.ieee.org/abstract/document/10801932)\]
- AV-Deepfake1M: A large-scale LLM-driven audio-visual deepfake dataset. \[[paper](https://dl.acm.org/doi/abs/10.1145/3664647.3680795)\]



# Attack on Speech-MLLMs

### Adversarial Speech Attacks and Exploits

- SpeechGuard: Exploring the adversarial robustness of multimodal large language models. \[[paper](https://aclanthology.org/2024.findings-acl.596/)\]
- Voice Jailbreak Attacks Against GPT-4o. \[[paper](https://arxiv.org/abs/2405.19103)\]

### Benchmark and Evaluation

- LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation. \[[paper](https://ieeexplore.ieee.org/abstract/document/10888070/)\]



# Defense Methods

### Visual Modality Defenses

- MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance. \[[paper](https://openreview.net/forum?id=5SDrx1CqmP)\]
- Mm-safetybench: A benchmark for safety evaluation of multimodal large language models. \[[paper](https://link.springer.com/chapter/10.1007/978-3-031-72992-8_22)\]
- Stop reasoning! when multimodal llms with chain-of-thought reasoning meets adversarial images. \[[paper](https://arxiv.org/abs/2402.14899)\]
- Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models. \[[paper](https://openreview.net/forum?id=WLPhywf1si)\]

### Audio and Speech Modality Defenses

- SpeechGuard: Exploring the adversarial robustness of multimodal large language models. \[[paper](https://aclanthology.org/2024.findings-acl.596/)\]
- TrustNavGPT: Modeling Uncertainty to Improve Trustworthiness of Audio-Guided LLM-Based Robot Navigation. \[[paper](https://ieeexplore.ieee.org/abstract/document/10801932/)\]

### Video Modality Defenses

- VideoQA in the Era of LLMs: An Empirical Study. \[[paper](https://link.springer.com/article/10.1007/s11263-025-02385-8)\]
- Beyond Raw Videos: Understanding Edited Videos with Large Multimodal Model. \[[paper](https://arxiv.org/abs/2406.10484)\]



# Contact

If you have any suggestions or find our work helpful, feel free to contact us

Email: {guijie,czjiang}@seu.edu.cn

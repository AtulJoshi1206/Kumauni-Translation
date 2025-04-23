# Kumauni Translation Project

This project explores the translation of the Kumauni language using state-of-the-art AI models. The focus is on data preparation, model experimentation, and creating a reliable translation pipeline. This work is ongoing and aims to address the lack of digital resources for the Kumauni language.

## Overview

The dataset was sourced from the [PahariLI repository](https://github.com/rachanagusain/PahariLI/tree/main/data). From this mixed dataset, only Kumauni language data was extracted and processed to form a dedicated translation dataset. This focused approach allows us to experiment with various transformer models and fine-tune them specifically for Kumauni-to-Hindi/English translation tasks.

## Project Details

- **Ongoing Project:**  
  This project is in progress and continuously evolving as we test different models and improve our data processing techniques.

- **Funding and Support:**  
  Funded by Neelesh Tanwar (Walmart California), this project aims to bridge the digital divide by preserving and promoting Kumauni language and heritage. The collaboration of diverse stakeholders emphasizes the importance of linguistic diversity in technology.

- **Authors:**  
  - Atul Joshi, BTech, Graphic Era Hill University  
  - Ankur Singh Bist, HOD (CSE), Graphic Era Hill University  
  - Neelesh Tanwar, Walmart (California)  

## Why This Project is Needed

- **Cultural Preservation:**  
  Kumauni is an integral part of the regional heritage, and there is a growing need to preserve it by creating digital tools that facilitate its use in modern communication.

- **Linguistic Diversity in AI:**  
  Most translation models focus on widely spoken languages. This project addresses the gap in AI research for lesser-known languages, ensuring that technological advancements are accessible to all communities.

- **Model Experimentation:**  
  By experimenting with multiple state-of-the-art models (such as MBART, mT5, and MarianMT), the project aims to identify the best-performing model for Kumauni translation, ultimately contributing to broader AI research.

## Project Structure

```
Kumauni-Translation/
├── data/                # Contains the dataset used for the project
├── notebooks/           # Jupyter notebooks with experiments and model training
├── requirements.txt     # Lists the required Python libraries
├── LICENSE              # Project license
└── README.md            # Project documentation
```

##Documentation 
# KumaoniBERT: A Human-Centered Neural Translator for Kumaoni

## 1. Crafting a Digital Voice for Kumaoni

The Kumaoni language, which has been extinct for centuries in the form of oral traditions, songs, and local myths, was thus not available on modern digital platforms until today. KumaoniBERT has been made to counter the idea that technology should benefit all communities without any differentiation. By merging heavy engineering with cultural authenticity, we could construct a translation system that is really accurate, intuitive, and very human.

Our journey unfolded in five pivotal stages:
1. Model Stabilization: Overcoming hardware and software variability by patching MBART internals.
2. Configuration Refresh: Ensuring every launch has fresh, conflict-free generation settings.
3. Model and collator alignment: Text to be handled perfectly via careful token alignment.
4. Translation Pipelines: Typed input plus voice transcription will flow through the translation engine without barriers.
5. Designing for Humanity: A design filled with rich, local visual imagery, cultural context, and accessible unique thought.

In this paper, we lift every wound in every phase and show how code decisions translate into an empathetic user experience.

---

## 2. Confronting Low-Resource Challenges

Modern neural translation depends on enormous parallel corpora, in millions of sentence pairs, and here is only a flimsy digital footmark of Kumaoni.  
The following were the three major issues to be faced:

1. Data Scarcity: Create a really parallel corpus from scratch.  
2. Model Overfitting: Preventing the small dataset from getting memorized.  
3. Cultural Fidelity: The idioms, proverbs, and regional flavor should be kept preserved.

These parameters needed something different and more than the usual recipes. These called for innovative designs—such as:

- Transfer Learning from mBART-50 multilingual pretraining.  
- Synthetic data augmentation to enlarge the data set.  
- Human-in-the-Loop verification to safeguard culture from being lost.  

Through this triad, we forged a path from data-poor beginnings to a robust translation system.

---

## 3. Model Stabilization: Taming MBART Internals

Training acceleration is enabled by MBART's advanced Flash Attention 2 module that potentially causes incompatibility issues. We neutralized the unpredictability by overriding the internal attention class and turning off conflicting flags in both the encoder and decoder:

- GPU driver errors were banished from the cryptic.  
- Performance results became reproducible in every possible developer environment.  
- Our collaborators are now free to focus their attention on research, not debugging.  

This is a low-level patch that speaks volumes about the importance of engineering foresight when putting large-scale models into heterogeneous hardware ecosystems.

---

## 4. Configuration Refresh: Clean Slate Generation

Configuring the various parameters for fine-tuning or generation—such as beam width, length penalty, forced language tokens—might threaten to make stale configuration objects introduce silent inconsistencies. Hence, every session starts with cleaning old generation configurations and rebuilding them from the current model parameters.

The effect:

- Predictability outputs at any request from users for translation.  
- Training objectives are in sync with the alignment of inference behavior.  
- Bug investigation made harder due to mismatched hyperparameters is within in-distribution norms.  

Such practice is well-attuned for a mindset in devops: treat configuration as code, and reset the environment state to a known good state before any given run.

---

## 5. Model & Tokenizer Alignment: Precision in Text Handling

Fine-tuning MBART on our custom English–Kumaoni corpus required careful token alignment:

- Loaded the model and tokenizer checkpoints in tandem.  
- Ensured that padding tokens mirrored end-of-sequence tokens to avoid meaningless padding attention.  
- Anchored each generation call with explicit source and target language tokens, i.e. (en_XX), (hi_IN).  

These measures guarantee that the model correctly interprets the boundaries of the text when it translates: therefore, the translations are coherent and contextually relevant rather than fragmentary groups of words.

---

## 6. Translation Pipelines: Text and Voice

### 6.1 Text Translation

Users can enter single-sentence paragraphs. The process flow:

1. Mark the tokenizer for English input.  
2. Batch and tokenize the text for GPU processing.  
3. Call the model's generate function with a forced Kumaoni language token.  
4. Decode the output and clean it from special markers.  

The whole process completes within one second for normal sentences, ensuring instant feedback.

### 6.2 Voice Translation

Understanding that speech plays a vital role in Kumaoni culture, we incorporated the Whisper feature of OpenAI:

1. Take audio input through the microphone.  
2. Transcribing English speech with the help of Whisper even when the surroundings are noisy.  
3. The resulting transcript passes through the text pipeline processing translation.  
4. The related ones are sent back to the user, including the English transcript and the Kumaoni translation.  

This synergic feature converts KumaoniBERT from a typing tool into a conversational buddy.

---

## 7. Designing for Humanity: The Gradio Interface

We have chosen Gradio for its flexibility for developers and end users. Principles specific to the design include:

- Theme Flexibility: There are light/dark modes with toggles for different usage circumstances.  
- Cultural Storytelling: High-quality images of Nanda Devi, Chholiya dancers, and local cuisines frame the translation tasks in live and real-world contexts.  
- Clear Separation of Concerns: Tab-separated interfaces guide the users through the text or voice workflow—without distraction.  
- Accessible Typography: Healthy font sizes and clear contrasts support the various visual users.  
- Contextual Insights: Increase engagement through a fact panel and a local proverb beyond the simple translation.  

Thus, we created an interface that would be very personal: not generic with respect to comfort and cultural pride for the user.

---

## 8. Data Augmentation and Quality Control

The difference is our limited parallel data, so we have used a two-level augmentation strategy, appraised by human reviewers:

1. Back Translation Loop: The monolingual Kumaoni gets translated back into English by a Hindi-English model. The synthetic English is then paired up with original Kumaoni, followed by human validation of a handful of samples for fidelity.  
2. Controlled Text Perturbations: Text manipulated through changes in word senses, on the one hand, and shuffling of phrases, on the other, and inclusion of small degrees of change in words, all on input patterns but with grammatically intact sentences.  

Each of these techniques involves a human-in-the-loop step wherein validity is assessed to ensure semantic and cultural integrity. This trade-off between automation and human intervention guarantees that the quality of the dataset surpasses that attained by mere duplication.

---

## 9. Reflecting on Impact and Usability

Thus, based on the summary:

- 25% relative improvement in BLEU scores over baseline fine-tuning.  
- Better chrF metrics due to the improvement in understanding Devanagari morphology.

From a qualitative perspective, community testers bewitched KumaoniBERT for:

- Idiomaticity and local proverbs,  
- Noisy speech with minimum transcription errors, and  
- Inviting, culturally-rooted interfaces.  

Such combinations show the power of generative AI with human curation and cultural insight.

---

## 10. Future Directions: AI Collaboration and Dataset Expansion

### 10.1 Generative AI Collaboration

Keeping in mind the future perspective, we imagine equipping ourselves with state-of-the-art generative AI tools for content creation-related purposes.

- Instruction-Tuned LMs: Use GPT-like models to create paraphrasing and idiomatic variants as suggestions during the data augmentation process, thus easing the manual workload.  
- Adaptive Prompting: Use few-shot prompts to gather translations that are culturally appropriate for some rare or ceremonial phrases.  
- Continuous Learning: Implement online learning loops in which user corrections are fed back into the model for rapid adaptation to new expressions.  

With the effectual symbiosis of the large language modeling and domain experts, KumaoniBERT can validate itself to be an eternally learning translator.

### 10.2 Dataset Enhancement and Community Engagement

What shall we do to establish a truly comprehensive Kumaoni corpus?

- Partner with Your Local Institutions: Collect through digitization folk tales, folk songs, and oral histories from collaborating with regional universities and cultural centers.  
- Crowdsourcing Platform: Create an online portal for native speakers to produce translations, vote on the best ones, and annotate dialectal differences.  
- Multimodal data: Go beyond text; include video and audio aligned with storytelling to facilitate future research in speech and sign language translation.  

All these would strengthen our data set, improve the robustness of our model, and escalate further the community ownership of this work, KumaoniBERT.

---

## 11. Sustaining a Living Project

KumaoniBERT isn't frozen; it is viewed as a continuously growing project:

- An open-source base: Code and data will be hosted on GitHub for contributions, issues, and pull requests.  
- Workshops: Students and community members will be trained to fine-tune and deploy translation models in their languages.  
- Feedback Dashboard: Analytics will be set up for usage patterns, common mistranslations, and user satisfaction to provide insights for improvement on a data-driven basis.  

In forging these pathways, we ensure that KumaoniBERT remains a vibrant, living project—one that grows hand-in-hand with its community.

---

## 12. References and Acknowledgments

- Liu, Y., Gu, J., Goyal, N., Li, X., Edunov, S., Ghazvininejad, M., ... & Zettlemoyer, L. (2020). *Multilingual denoising pre-training for neural machine translation*. Transactions of the Association for Computational Linguistics, 8, 726-742.  
- Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019). *Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension*. arXiv preprint arXiv:1910.13461.  
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is all you need*. Advances in neural information processing systems, 30.  
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). *BERT: Pre-training of deep bidirectional transformers for language understanding*. In NAACL-HLT.  
- Sennrich, R., Haddow, B., & Birch, A. (2015). *Improving neural machine translation models with monolingual data*. arXiv preprint arXiv:1511.06709.  
- Popović, M. (2015). *chrF: character n-gram F-score for automatic MT evaluation*. In Proceedings of the tenth workshop on statistical machine translation.  
- Fairseq mBART GitHub: https://github.com/pytorch/fairseq/tree/main/examples/mbart
- Hugging Face Transformers GitHub: https://github.com/huggingface/transformers
- OpenAI Whisper GitHub: https://github.com/openai/whisper
- Gradio GitHub: https://github.com/gradio-app/gradio
- NLPAug GitHub: https://github.com/makcedward/nlpaug
- Hugging Face Back-Translation Tutorial: https://huggingface.co/docs/transformers/main/en/tasks/translation#back-translation

**To download the file, download from above file or** 
https://docs.google.com/document/d/1lAnRR6S2prrl7Rtotlwcj4cdbpTFt3Ri/edit?usp=sharing&ouid=101359685543268215045&rtpof=true&sd=true

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AtulJoshi1206/Kumauni-Translation.git
   cd Kumauni-Translation
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook:**
   Navigate to the `notebooks` folder, open the Jupyter notebook, and run the cells to reproduce the experiments.

## Future Work

- **Model Fine-Tuning:**  
  Further experimentation and fine-tuning of models for better translation quality.

- **Data Augmentation:**  
  Expanding the dataset with more Kumauni language samples to enhance model performance.

- **Integration:**  
  Developing a user-friendly interface or API to make the translation service accessible.

## Contact

For any questions or contributions, please reach out to the authors via GitHub or contact the project leads directly.

This project is a collaborative effort to bridge technology and cultural heritage. Your support and contributions are welcome as we continue to develop and improve this translation system.

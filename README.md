# Generalizable Hearthstone Agent
This repository is the official implementation of *Language-Guided Value with Latent Dynamics for Textual Card Game Agent*.

This work presents a novel framework that integrates large language models (LLMs) with reinforcement learning (RL) agents to improve training efficiency in textual card games like Hearthstone. By leveraging a fine-tuned T5-base model to encode and interpret card strategies expressed in natural language, the framework yields a language-guided value function that computes Q-values with language embeddings. An auxiliary transition loss in the latent space further enhances the agent's ability to capture complex game dynamics, facilitating efficient policy learning across a large set of cards and decks.

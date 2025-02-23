# -*- coding: utf-8 -*-
"""
Multi-Agent Debate Experiment Code
Objective: Improve the accuracy of language models in reasoning tasks through multi-agent debate.
"""

# Import necessary libraries
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import random

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-base"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load the GSM8K dataset
dataset = load_dataset("gsm8k", split="train[:100]")  # Use a subset of the dataset for testing


# Define the multi-agent debate environment
class MultiAgentDebate:
    def __init__(self, model, tokenizer, num_agents=2, num_rounds=3):
        self.model = model
        self.tokenizer = tokenizer
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.agents = [f"Agent {i+1}" for i in range(num_agents)]

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def run_debate(self, question):
        print(f"Question: {question}")
        debate_context = question
        for round in range(self.num_rounds):
            print(f"\nRound {round + 1}")
            for agent in self.agents:
                prompt = f"{debate_context}\n{agent}: "
                response = self.generate_response(prompt)
                print(f"{agent}: {response}")
                debate_context += f"\n{agent}: {response}"
        return debate_context


# Define the evaluation function for reasoning tasks
def evaluate_model(question, answer):
    # A simple evaluation function: check if the model's generated response contains the correct answer
    response = agent.generate_response(question)
    return answer.lower() in response.lower()


def evaluate_debate(question, answer, num_agents, num_rounds):
    debate_env = MultiAgentDebate(model, tokenizer, num_agents=num_agents, num_rounds=num_rounds)
    debate_context = debate_env.run_debate(question)
    return answer.lower() in debate_context.lower()


# Randomly select a sample question
sample = random.choice(dataset)
question = sample['question']
answer = sample['answer']

print(f"\nSample Question: {question}")
print(f"Correct Answer: {answer}")

# Single-agent reasoning
agent = MultiAgentDebate(model, tokenizer, num_agents=1, num_rounds=1)
single_agent_accuracy = evaluate_model(question, answer)
print(f"\nSingle Agent Accuracy: {single_agent_accuracy}")

# Multi-agent debate
num_agents = 3
num_rounds = 3
debate_accuracy = evaluate_debate(question, answer, num_agents, num_rounds)
print(f"\nDebate Accuracy (Agents: {num_agents}, Rounds: {num_rounds}): {debate_accuracy}")
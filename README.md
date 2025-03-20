![(title)](title.png)

![Python](https://img.shields.io/badge/Python-3.10-blue) [![Medium](https://img.shields.io/badge/Medium-Read%20Now-red?logo=medium)](https://medium.com/@fareedkhandev) ![Contributions](https://img.shields.io/badge/Contributions-Welcome-yellow) 


The entire training process of DeepSeek R1 is nothing but using different way of reinforcement learning on top of their base model (i.e. [deepseek V3](https://huggingface.co/deepseek-ai/DeepSeek-V3))

To make everything easy to understand we will use hand drawn flowcharts along with the code and will follow the step by step implementation of deepseek technical report and will build our own model using a tiny base model that you can also run locally.

We will also be covering theory next to each step, so in case you are not aware of DeepSeek R1 this blog will cover that too.

I explained **DeepSeek R1** with **hand-drawn diagrams** for non-tech folks. [Read here](https://levelup.gitconnected.com/drawing-deepseek-r1-architecture-and-training-process-from-scratch-72043da33955).

<!-- omit in toc -->
## GitHub Code Overview

The codebase is organized as follows:

```bash
train-deepseek-r1/
├── code.ipynb         # Jupyter notebook with code implementation
├── requirements.txt   # List of required libraries
└── r1_for_dummies.md  # DeepSeek R1 explanation for non-tech folks
```

<!-- omit in toc -->
## Table of Contents
- [配置开发环境](#配置开发环境)
- [训练数据集](#训练数据集)
- [DeepSeek R1 训练概览](#DeepSeek-R1-训练概览)
- [选择基础模型](#选择基础模型)
- [强化学习中的策略模型(R)](#policy-model-r-in-rl-setup)
- [R1 Zero中的GRPO算法](#grpo-algorithm-for-r1-zero)
- [提示词模板](#prompt-template)
- [对训练数据进行预处理](#preprocessing-training-data)
- [奖励函数](#reward-functions)
  - [准确性奖励](#accuracy-reward)
  - [格式奖励](#format-reward)
  - [推理步骤奖励](#reasoning-steps-reward)
  - [余弦缩放奖励(Cosine Scaled Reward)](#cosine-scaled-reward)
  - [重复性惩罚奖励(Repetition Penalty Reward)](#repetition-penalty-reward)
- [R1 Zero的训练配置](#training-configurations-for-r1-zero)
- [GRPO训练循环](#grpo-training-loop)
- [保存 Tiny R1 Zero LLM](#saving-tiny-r1-zero-llm)
- [R1 Zero的两个主要问题](#two-main-problems-with-r1-zero)
- [为SFT准备冷启动数据](#preparing-cold-start-data-for-sft)
- [使用长思维链(Long CoT)进行少样本(few-shot)提示](#few-shot-prompting-with-long-cot)
- [直接进行提示](#direct-prompting)
- [后处理细化(Post Processing Refinement)](#post-processing-refinement)
- [使用冷启动数据进行第一阶段的SFT](#sft-stage-1-with-cold-start-data)
- [Stage 1 SFT Trainer Configs for R1](#stage-1-sft-trainer-configs-for-r1)
- [Stage 1 STF Training Loop](#stage-1-stf-training-loop)
- [保存 Tiny R1 LLM 模型](#saving-tiny-r1-llm)
- [面向推理的强化学习](#reasoning-oriented-rl)
- [拒绝采样(Rejection Sampling)](#rejection-sampling)
- [第二阶段的SFT训练](#sft-stage-2-training)
- [蒸馏(Distillation)](#distillation)



## 配置开发环境

使用以下命令克隆存储库并安装所需的库：

```bash
git clone https://github.com/confucianzuoyuan/train-deepseek-r1.git
cd train-deepseek-r1
pip install -r requirements.txt
```

现在，让我们导入所需的库并为我们的训练设置环境。

```python
# 导入必要的库
import logging
import os
import sys
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

# 导入PyTorch和Hugging Face Transformers库
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint

# 导入数据集相关库
import datasets
from datasets import load_dataset

# 从 TRL (Transformers Reinforcement Learning) 导入相关库
from trl import (
    AutoModelForCausalLMWithValueHead, 
    PPOConfig, 
    PPOTrainer, 
    GRPOTrainer, 
    GRPOConfig, 
    SFTTrainer
)

# 导入数学相关的库
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
```

## 训练数据集

虽然论文没有说明 RL 预训练所使用的初始数据集，但我们假设它应该是推理方面的数据集。

因此为了尽可能接近原始复制，我们将使用这两个开源的推理数据集（来自Hugging Face）：

 1. [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (训练 R1 Zero 时使用)

 2. [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) (训练 R1 时使用)

AI-MO/NuminaMath-TIR 包含 70K 个数学问题，其中的messages列显示了解答背后的 COT（思维链）推理。

| Field    | Description |  
|----------|------------|  
| problem  | The math problem |  
| solution | Step-by-step solution |  
| messages    | Chat to solve the problem |

看一下数据集的样本：

```python
# 从 DigitalLearningGmbH 加载 "AI-MO/NuminaMath-TIR" 数据集
MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")  

# 获取训练数据的第一条数据（样本）
MATH_le['train'][0]


#### OUTPUT ####
{
'problem': 'What is the degree of the polynomial 4 +5x^3 ... ',
'solution': 'This polynomial is not written in ...',
'messages': [{'from': 'user', 'value': 'The problem ...'}]
}
#### OUTPUT ####
```

而 Bespoke-Stratos 数据集包含 17K 个专注于数学和代码的问题。

| Field        | Description |  
|-------------|------------|  
| system      | Guidelines for math and code problems |  
| conversation | Chat to solve the problem |

它的数据样本如下所示：

```python
# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default") 

# Access the first sample in the training set
bespoke_rl['train'][0]


#### OUTPUT ####
{
'system': 'Your role as an assistant involves ... ',
'conversations': [{'from': 'user', 'value': 'Return your ...'}]
}
 #### OUTPUT ####
```

你不一定要选择这两个数据集，可以选择任何一个面向推理的数据集（**包含问题及问题的分步解答**）。

## DeepSeek R1 训练概览

因此，在介绍技术实现之前，我们需要明白 DeepSeek-R1 并非从头开始训练的，也就是说，从零开始训练。相反，他们从一个非常聪明的模型开始，而他们已经有了[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) LLM，但他们想让它成为推理届的超级明星。

![DeepSeek R1 Implementation Quick Overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5872/1*XhE5MVuPfOrrbBxgYtHMsg.png)

To do that, they used **Reinforcement Learning**, or RL for short where you reward the LLM when it does something good for reasoning while punish it otherwise.

为了做到这一点，他们使用了**强化学习**（简称 RL），当 LLM 输出有益于推理的响应时，就奖励 LLM ，否则就惩罚 LLM 。

但这不仅仅是一个简单的训练环节。它就像是一大堆步骤，他们称之为流水线。他们首先尝试了纯强化学习 ，看看推理是否会自行出现，这就是 **DeepSeek-R1-Zero** ，有点像一个实验。然后对于真正的 **DeepSeek-R1** ，他们通过不同的阶段使其更有条理。他们给它一些启动数据让它运行，然后进行强化学习，然后是更多的数据，然后是更多的强化学习……就像是一步步升级！

关键在于让这些语言模型更好地思考问题。

> 是的，在我们深入研究每个步骤的疯狂细节之前，这是非常简短的版本

## 选择基础模型

由于 DeepSeek 团队选择了 DeepSeek-V3 作为基础模型来创建 R1 Zero 和 R1，但它的大小相当庞大（**685 GB💀**），显然超出了我们的承受范围。

为简单起见，我们将使用小得多的基础模型 [Qwen/Qwen2.5–0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（大小为 0.9 GB）。如果有更大的 GPU RAM，甚至可以加载未量化的 LLM，那么可以选择更大的模型，例如 [Qwen/Qwen2.5–7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) 。

让我们看一下我们用的基础模型的一些规格：

```python
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training" # For saving our trained model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer with chat template
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


#### OUTPUT ####
Vocabulary size: 151665
Model max length: 131072
Pad token: <|endoftext|>
EOS token: <|im_end|>
#### OUTPUT ####
```

这些是有关模型的一些基本信息，请查看我们的基础模型的参数总数。

```python
# Initialize base model
model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  trust_remote_code=True,
  torch_dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")


#### OUTPUT ####
Model parameters: 494,032,768
#### OUTPUT ####
```

接近 0.5B 个参数，让我们从中打印一个简单的响应，然后我们将继续下一步。

```python
# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the appropriate device
model.to(device)

# Test basic inference
def test_model_inference(user_input: str):
    """Test basic model inference with the loaded model and tokenizer."""
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and generate
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")


#### OUTPUT ####
"Test Input: how are you?
Model Response: As an AI language model I dont have feelings ..."
#### OUTPUT ####
```

所以，这个小的大模型的输出非常可靠，并且肯定适合我们去训练和 DeepSeek 相似的模型。

## Policy Model (R) In RL Setup

Now that we have selected our base model, next we need to understand how a basic RL setup works for training an LLM.

现在我们已经选择了基础模型，接下来我们需要了解基本 RL 设置如何训练 LLM。

For DeepSeek R1 their starting point was (DeepSeek V3) base model and in our case we are starting with Qwen2.5–0.5B-Instruct. By a starting point I meant that **it has created the DeepSeek R1 zero version**, an initial version which has some errors in it before the final version was created.

对于 DeepSeek R1，他们的起点是（DeepSeek V3）基础模型，而在我们的案例中，我们从 Qwen2.5–0.5B-Instruct 开始。我所说的起点是指 **它已经创建了 DeepSeek R1 zero版本** ，这是在创建最终版本之前包含一些错误的初始版本。

The initial version (R1 Zero) was created using Reinforcement Learning where (DeepSeek v3/Qwen2.5–0.5B) acts as an RL agent (actor who takes action). Let’s first visualize how it works.

初始版本 (R1 Zero) 是使用强化学习创建的，其中 (DeepSeek v3/Qwen2.5–0.5B) 充当强化学习的agent（采取行动的参与者）。让我们首先直观地了解一下它的工作原理。

![Qwen 2.5 as an agent workflow (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5410/1*S6YIXu1vIVmQFl-DgRFktg.png)

The RL agent (DeepSeek V3/Qwen2–0.5B) starts by taking an **Action**, which means it generates an answer and some reasoning for a given problem that’s put into its **Environment**. The Environment, in this case, is simply the reasoning task itself.

强化学习Agent (DeepSeek V3/Qwen2–0.5B) 首先采取一个 **Action** ，这意味着它会针对给定的问题生成答案和一些推理，并将其放入其 **Environment** 中。在这种情况下，Environment 就是推理任务本身。

After taking an action, the Environment gives back a **Reward**. This Reward is like feedback, it tells our base model (DeepSeek V3/Qwen2–0.5B) how good its action was. A positive Reward means it did something right, maybe got the answer correct or reasoned well. This feedback signal then goes back to our base model, helping it learn and adjust how it takes actions in the future to get even better Rewards.

采取 Action 后，Environment会给出Reward。这个Reward就像反馈，它告诉我们的基础模型（DeepSeek V3/Qwen2–0.5B）它的Action有多好。积极的Reward意味着它做对了某件事，可能得到了正确的答案或推理得很好。这个反馈信号随后会返回到我们的基础模型，帮助它学习和调整未来如何采取Action以获得更好的Reward。

> 在下一节中，我们将更详细地讨论这种方法

## GRPO Algorithm for R1 Zero

So that we have understand a basic RL flow now we need to learn what exact RL algorithm DeepSeek uses for R1-Zero.

为了理解基本的 RL 流程，现在我们需要了解 DeepSeek 对 R1-Zero 使用的具体 RL 算法。

There are many RL algos available, but traditional RL use something called a **“critic” **to help the main decision making part (“actor” i.e. DeepSeek-V3/Qwen2-0.5B). This critic is usually just as big and complex as the actor itself, which basically doubles the amount of computational cost.

有许多可用的 RL 算法，但传统的 RL 使用一种称为 **"critic"** 的东西来帮助主要决策部分（“actor”，即 DeepSeek-V3/Qwen2-0.5B）。这个critic通常与actor本身一样大和复杂，这基本上使计算成本翻倍。

But DeepSeek uses GRPO for training their initial (R1 Zero), **GRPO** does things differently because it figures out a baseline, a kind of reference point for good actions directly from the results it gets from a **group** of actions. Because of this, GRPO doesn’t need a separate critic model at all. This saves a lot of computation and makes things more efficient.

但是 DeepSeek 使用 **GRPO** 来训练其初始模型（R1 Zero）， GRPO 的做法有所不同，因为它会根据一组action的结果直接找出基线，即良好action的参考点。因此，GRPO 根本不需要单独的critic模型。这节省了大量计算并提高了效率。

Let’s draw a flowchart of how GRPO is being used for R1 Zero training, and then we will **interpretate** it.

让我们绘制一个流程图，说明如何将 GRPO 用于 R1 Zero 的训练，然后对其进行 **解释** 。

![GRPO Flow for DeepSeek R1 Zero (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6404/1*8mfNzi-gvasR7mSaseswmg.png)

Let’s understand how DeepSeek GRPO implementation works with our base model (Qwen2–0.5B). 

让我们了解一下 DeepSeek 的 GRPO 实现如何与我们的基础模型（Qwen2-0.5B）协同工作。

First, the **Problem Input (A)** is given to the **Qwen Model (B)**, Qwen attempts to generate an answer through **Generate Completion (C)**. The final result, called the **Completion Output (D)**, includes reasoning steps in <think> tags and the final solution in <answer> tags.

首先， **问题输入（A）** 被输入到 **Qwen 模型（B）** 中，Qwen 尝试通过 **生成补全（C）** 来产生答案。最终结果称为 **完成输出（D）** ，其中包括 `<think>` 标签中的推理步骤和 `<answer>` 标签中的最终解决方案。

Next, the **Problem Input (A)** and the **Ground Truth Solution (E)** are fed into the **Reward Functions (F)**, acting as intelligent graders. These functions compare Qwen **Completion Output (D)** with the correct solution and evaluate different aspects such as:

接下来， **问题输入 (A)** 和 **真实答案 (E)** 被输入到 **Reward函数 (F)** 中，充当智能评分器。这些函数将 Qwen 的 **完成输出(D)** 与正确答案进行比较，并评估不同方面，例如：

 1. **准确性** (答案正确吗？)

 2. **格式** (`<think>` 和 `<answer>` 标签是否正确使用了？)

 3. **推理步骤** (逻辑清楚吗?)

 4. **余弦缩放(Cosine Scaling)** (响应是否简洁？)

 5. **重复性惩罚(Repetition Penalty)** (是否存在不必要的重复？).

These evaluations produce **Reward Scores (G)**, which are then passed to the **GRPO Trainer (H)**. The trainer uses gradients to adjust the **Qwen Model (B)**, fine-tuning how it generates answers. This process is called **Gradient Reward Policy Optimization** because it optimizes Qwen responses using **gradients**, **reward feedback**, and **policy adjustments** to maximize performance.

这些评估会产生 **Reward分数 (G)** ，然后传递给 **GRPO训练器(H)** 。训练器使用梯度来调整 **Qwen模型(B)** ，微调其生成答案的方式。这个过程被称为 **Gradient Reward Policy Optimization**，因为它使用 **gradient** 、**reward** 反馈 和 **policy** 调整来优化 Qwen 响应，以最大限度地提高性能。

Finally, the updated **Qwen Model (B)** is tested again on new problems, continuously refining itself through repeated cycles. With each iteration, Qwen becomes a better problem solver.

最后，更新后的 **Qwen模型(B)** 再次在新问题上进行测试，通过反复循环不断完善自身。随着每次迭代，Qwen 都会成为更好的问题解决者。

> 在下一节中，我们将开始预处理 GRPO 训练所使用的训练数据集

## Prompt Template

We are using the same thinking prompt template that DeepSeek uses for the GRPO algorithm to build R1 Zero, so let’s define that:

我们使用与 DeepSeek 用于 GRPO 算法的相同思维提示模板来构建 R1 Zero，因此让我们定义：

```python
# DeepSeek system prompt for GRPO based training
SYSTEM_PROMPT = (
  f"""A conversation between User and Assistant. The user asks a question, 
      and the Assistant solves it. The assistant
      first thinks about the reasoning process in the mind and 
      then provides the user with the answer. The reasoning
      process and answer are enclosed within <think> </think> 
      and <answer> </answer> tags, respectively, i.e., 
      <think> reasoning process here </think><answer> answer here </answer>
   """
)
```

This **system prompt** tells the base model (Qwen2–0.5B) its role as a helpful assistant who reasons step-by-step before answering.

该系统提示告诉基础模型（Qwen2-0.5B）它的角色是作为一个有用的助手，在回答之前逐步进行推理。

The `<think>` and `<answer>` tags are used to structure the model response, separating its internal reasoning from the final answer for better evaluation and reward.

`<think>` 和 `<answer>` 标签用于构建模型响应，将其内部推理与最终答案分开，以便更好地评估和奖励。

## Preprocessing Training Data

Now that we have our system prompt ready, we need to transform our training data according to our template.

现在我们已经准备好系统提示，我们需要根据模板转换训练数据。

![Preprocessing dataset overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6160/1*XnM7v4dPD4LtyAh2MLuInA.png)

We need to create the make_conversation function that will handle the conversation for us.

我们需要创建 `make_conversation` 函数来为我们处理对话。

```python
# Function to structure the training data
def make_conversation(example):
  """Convert dataset examples into conversation format."""
  return {
      "prompt": [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": example["problem"]},
      ],
  }
```

It will take each problem column value from our training dataset and return a dictionary with the system prompt and the appended problem question for each row. Let’s create this function that will prepare our dataset.

它将从我们的训练数据集中获取每个问题列的值，并返回一个包含系统提示和每行附加问题问题的字典。让我们创建这个函数来准备我们的数据集。

```python
# Load and prepare dataset
def load_math_dataset():
    """Load and prepare the mathematics dataset."""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # Convert splits into dictionary
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # Apply conversation format
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # Remove 'messages' column if exists
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset
```

We have everything ready, let’s transform our training data into the required format and print the training and test size.

我们已经准备好一切，让我们将训练数据转换为所需的格式并打印训练和测试规模。

```python
# Load our training dataset and printing train/test size
dataset = load_math_dataset()

print(f"Train set size: {len(dataset['train'])}")
print(f"Test set size: {len(dataset['test'])}")



#### OUTPUT ####
Train set size: 72441
Test set size: 99
#### OUTPUT ####
```

Now that we have split our training dataset, we need to validate our dataset (**Check if user/assistant conversation exist**) before moving to the next step.

现在我们已经分割了训练数据集，在进入下一步之前，我们需要验证数据集（**检查用户/助手对话是否存在**）。

```python
def validate_dataset(dataset):
    """Perform basic validation checks on the dataset."""
    
    # Define the required fields for the dataset
    required_fields = ["problem", "prompt"]

    # Loop through the 'train' and 'test' splits of the dataset
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")

        # Retrieve column names from the dataset
        fields = dataset[split].column_names

        # Check if any required fields are missing
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"Warning: Missing fields: {missing}")  # Warn if fields are missing
        else:
            print("✓ All required fields present")  # Confirm all fields are present

        # Retrieve the first sample from the dataset split
        sample = dataset[split][0]

        # Extract the 'prompt' field, which contains a list of messages
        messages = sample['prompt']

        # Validate the prompt format:
        # - It should contain at least two messages
        # - The first message should be from the 'system' role
        # - The second message should be from the 'user' role
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ Prompt format is correct")  # Confirm correct format
        else:
            print("Warning: Incorrect prompt format")  # Warn if format is incorrect

# Validate dataset
validate_dataset(dataset)
```

输出如下：

```
Validating train split:

✓ All required fields present
✓ Prompt format is correct

Validating test split:

✓ All required fields present
✓ Prompt format is correct
```

Our training dataset is validated successfully 🙌, it means we have successfully transformed our dataset for training.

我们的训练数据集已成功验证🙌，这意味着我们已成功转换数据集以进行训练。

## 奖励函数

We already saw in GRPO section that it evaluate the answer of base model through five different ways:

我们已经在 GRPO 部分看到，它通过五种不同的方式评估基础模型的答案：

![Reward Functions (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7474/1*kJln8i6Tv4aspnTfMoRW-Q.png)

 1. **准确性** (答案正确吗？)

 2. **格式** (`<think>` 和 `<answer>` 标签是否正确使用了？)

 3. **推理步骤** (逻辑清楚吗?)

 4. **余弦缩放(Cosine Scaling)** (响应是否简洁？)

 5. **重复性惩罚(Repetition Penalty)** (是否存在不必要的重复？).

Each of these are functions will calculate the reward for each response, and we need to code them. So, let’s do that first.

这些函数都会计算每个响应的奖励，我们需要对它们进行编码。所以，让我们先这样做。

### 准确性奖励

Accuracy reward is the most easy to understand but requires a bit complex code. In this reward model we want to check if mathematically our base model response is equivalent to the ground truth solution.

准确率奖励最容易理解，但需要稍微复杂的代码。在这个奖励模型中，我们想要检查从数学上讲我们的基础模型响应是否等同于真实答案。

![Accuracy Reward (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7860/1*A3tW-OZSZ4m10EEzogjy8Q.png)

If the model answer is mathematically correct, we assign a reward of **1.0**. If it is incorrect, the reward is **0.0**. In cases where the ground truth solution cannot be parsed, we assign a neutral reward of **0.5** to avoid unfair penalties.

如果模型答案在数学上是正确的，我们将分配 **1.0** 的奖励。如果不正确，则奖励为 **0.0** 。在无法解析基本事实解决方案的情况下，我们将分配 **0.5** 的中性奖励，以避免不公平的惩罚。

Now, let’s implement the function.

现在，让我们实现该功能。

```python
def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    
    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        # Parse the ground truth solution
        gold_parsed = parse(sol, extraction_mode="first_match", 
                            extraction_config=[LatexExtractionConfig()])
        
        if gold_parsed:  # Check if parsing was successful
            # Parse the model's answer with relaxed normalization
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            # Reward 1.0 if correct, 0.0 if incorrect
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If ground truth cannot be parsed, assign neutral reward (0.5)
            reward = 0.5
            print("Warning: Failed to parse gold solution:", sol)

        rewards.append(reward)
    
    return rewards
```

In this function, we check whether the model response is **equivalent** to the correct answer. Instead of comparing raw text, we:

在此函数中，我们检查模型响应是否等同于正确答案。我们不比较原始文本，而是：

 1. Convert the solution into a structured mathematical format using **latex2sympy2** 使用 **latex2sympy2** 将答案转换为结构化数学格式。

 2. If parsing fails, assign a neutral reward of **0.5** 如果解析失败，则分配 **0.5** 的中性奖励。

 3. Extract the model output and normalize it for better robustness 提取模型输出并进行规范化以获得更好的鲁棒性。

 4. Use **math_verify** to check if the parsed response matches the parsed solution 使用 **math_verify** 检查解析的响应是否与解析的解决方案匹配。

 5. If correct assign **1,** if incorrect assign **0** 如果正确则分配 **1** ，如果不正确则分配 **0** 。

This ensures that accuracy evaluation is not just about textual similarity but **true mathematical correctness.**

这确保了准确性评估不仅仅涉及文本相似性，还涉及 **真正的数学正确性** 。

### 格式奖励

Format Reward is all about making sure our model follows instructions and structures its output correctly. We asked it to put its reasoning in `<think>` tags and the final answer in `<answer>` tags, right? This reward function checks exactly that!

格式奖励就是确保我们的模型遵循指令并正确构建其输出。我们要求它将推理放在 `<think>` 标签中，将最终答案放在 `<answer>` 标签中，对吗？此奖励函数正是检查这一点！

![Forward Reward (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6620/1*DbUraziwiOoAj6SvtSJmpw.png)

If the model uses those tags correctly, we give it a reward of 1. If it messes up the format, it gets 0. Simple as that! This encourages the model to pay attention to the output structure we want.

如果模型正确使用了这些标签，我们会给它 **1** 的奖励。如果格式混乱，就会得到 **0** 。就这么简单！这鼓励模型关注我们想要的输出结构。

让我们编程实现:

```python
# Implement Format Reward Function
def format_reward(completions, **kwargs):
  """
  Reward function to check if the completion has the correct format:
  <think>...</think> <answer>...</answer>.
  """
  # Define the regex pattern for the desired format
  pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

  # Extract the content from each completion
  completion_contents = [completion[0]["content"] for completion in completions]

  # Check if each completion matches the pattern
  matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
             for content in completion_contents]

  # Reward 1.0 for correct format, 0.0 otherwise
  return [1.0 if match else 0.0 for match in matches]
```

在这个函数里:

* We define a pattern using regular expressions (regex). This pattern basically says “the content should *start* with <think>, have *anything* inside until </think>, then some *spaces*, then <answer>, *anything* inside until </answer>, and then *end* there”.

* 我们使用正则表达式 (regex) 定义一个模式。该模式的基本含义是“内容应以开头，其中包含任何内容直到 ，然后是一些空格，然后是 ，其中包含任何内容直到 ，然后结束于 ”。

* We get the actual text content from each model completion.

* 我们从每个模型完成中获取实际的文本内容。

* Then we use use re.match to see if each content perfectly matches our pattern. re.DOTALL helps the . in regex match newlines too, and re.MULTILINE makes ^ and $ match the start/end of the whole string, not just lines.

* 然后我们使用 `re.match` 来查看每个内容是否完全匹配我们的模式。`re.DOTALL` 帮助正则表达式中的 . 匹配换行符，而 `re.MULTILINE` 使 `^` 和 `$` 匹配整个字符串的开始/结束，而不仅仅是行。

* Finally, we give a reward 1 if it matched the format perfectly, 0 if it didn’t. This is a strict on/off reward for format correctness.

* 最后，如果格式完全匹配，我们会给予奖励 **1** ，如果不匹配，则会给予奖励 **0** 。这是对格式正确性的严格开/关奖励。

### Reasoning Steps Reward

Reasoning Steps Reward is a bit clever. We want to encourage our model to show its **“thinking process”**. So, we are going to reward it for including things that *look like* reasoning steps.

推理步骤奖励有点聪明。我们想鼓励我们的模型展示它的 **“思考过程”** 。因此，我们将奖励它包括看起来像推理步骤的内容。

![Reasoning Steps Reward Encouragement (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/5406/1*hx0sAVnY58WOYw6rGF64ug.png)

We will look for keywords and patterns that usually show up in step-by-step reasoning, like:

我们将寻找在逐步推理中通常出现的关键词和模式，例如：

* Step 1, Step 2, etc. 步骤 1、步骤 2 等等。

* Numbered lists like 1, 2 编号列表，如 1、2

* Bullet points like - or * 项目符号如 `-` 或 `*`

* Transition words like First, Second, Next, Finally 过渡词，如“第一”、“第二”、“下一步”、“最后”

The more of these it includes, the better the reward. It’s like giving points for showing its work!

它包含的内容越多，奖励就越好。这就像展示它的工作而给予积分一样！

Let’s code this reasoning encouraging function:

让我们编写这个推理鼓励函数：

```python
def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]
```

We create a pattern that’s a bit more complex regex. It looks for all those reasoning indicator things we listed above.

我们创建一个稍微复杂一点的正则表达式模式。它会查找我们上面列出的所有推理指标。

We use re.findall to find *all* the matches of our pattern within each content. `len(re.findall(…))` then gives us the *count* of these indicators.

我们使用 re.findall 在每个内容中查找符合我们模式的所有匹配项。len len(re.findall(…))然后为我们提供这些指标的数量。

The reward is calculated as min(1.0, count / 3). This means

奖励的计算方式为 min(1.0, count / 3)。这意味着

* If it finds 3 or more reasoning indicators ( count >= 3), the reward is 1.0 (max reward).

* 如果它发现 3 个或更多推理指标（计数 >= 3），则奖励为 1.0（最大奖励）。

* If it finds fewer (e.g., count = 1 or 2), it gets a *partial* reward (like 1/3 or 2/3).

* 如果发现较少的数量（例如，count = 1 或 2），它会获得部分奖励（如 1/3 或 2/3）。

* If it finds none (count = 0), the reward is 0.0.

* 如果没有找到（计数 = 0），则奖励为 0.0。

The / 3 is a bit of a magic number here. We’re saying **“aim for about 3 reasoning steps to get full credit”** You can tweak this number if you want to encourage more or fewer steps.

`/ 3` 是一个神奇的数字。我们说“目标是完成大约 3 个推理步骤才能获得满分”如果您想鼓励更多或更少的步骤，您可以调整这个数字。

### Cosine Scaled Reward

Cosine Scaled Reward is a bit more advanced. It’s about encouraging *conciseness* in correct answers and being *less harsh* on longer incorrect answers.

余弦缩放奖励稍微高级一些。它鼓励回答简洁的正确答案，对较长的错误答案则不那么苛刻。

![Cosine Scaling Concept (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7094/1*WmG8r1OVeU4R3jObAy0yCg.png)

Think of it like this:

* **For correct answers:** We want to reward *shorter*, more direct solutions more than long, rambling ones. A short, correct answer is often better.

* 对于正确答案：我们希望奖励更简短、更直接的解决方案，而不是冗长、漫无目的的解决方案。简短、正确的答案通常更好。

* **For incorrect answers:** A short, wrong answer is probably worse than a longer, wrong answer that at least *tried* to reason. So, we want to penalize short wrong answers *more* than long wrong answers.

* 对于错误答案：简短的错误答案可能比至少尝试推理的较长的错误答案更糟糕。因此，我们希望对简短的错误答案的惩罚比对较长的错误答案的惩罚更大。

Let’s see the code that does this clever scaling:

让我们看看实现这种巧妙缩放的代码：

```python
# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completions, solution, accuracy_rewards, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solution, accuracy_rewards):
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len # How far we are to max length
            cosine = math.cos(progress * math.pi) # Cosine value based on progress

            if acc_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else: # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward
```

`get_cosine_scaled_reward(...)` generates a reward function for training, customizing scaling with parameters like min_value_wrong/max_value_wrong (penalty range for incorrect answers) and min_value_correct/max_value_correct (reward range for correct ones). max_len sets the maximum length for scaling.

`get_cosine_scaled_reward(...)` 生成一个用于训练的奖励函数，使用 `min_value_wrong/max_value_wrong`（错误答案的惩罚范围）和 `min_value_correct/max_value_correct`（正确答案的奖励范围）等参数定制缩放。`max_len` 设置缩放的最大长度。

Inside, `cosine_scaled_reward(...)` we calculate rewards based on completions, solution, and accuracy_rewards.

It computes gen_len, normalizes it as progress `= gen_len / max_len`, and derives a cosine value that starts at 1 (short answers) and decreases to -1 (long answers).

If `acc_reward > 0.5`, it uses the correct reward range, otherwise it applies the incorrect range but swaps min/max values to penalize longer wrong answers less.

### Repetition Penalty Reward

Repetition Penalty Reward is all about discouraging our model from getting stuck in loops and repeating itself. We want it to generate fresh, varied reasoning and answers, not just copy-paste the same phrases over and over!

![Repetition Penalty Idea (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/8608/1*9jBhiz-rI_fRGa77g9RZtQ.png)

This reward function penalizes the model if it uses the same sequences of words (n-grams) too many times. We’ll use n-grams of size 3 (trigrams) in our example, but you can adjust this.

If the model repeats itself a lot, it gets a negative reward (penalty). If it’s more diverse and avoids repetition, the penalty is less.

Let’s implement the code to penalize repetition:
```python
def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # Generate n-grams
                ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
                total += 1 # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return get_repetition_penalty_reward
```

Our `get_repetition_penalty_reward(...)` creates a reward function to penalize repetition, with parameters like ngram_size (default 3, for trigrams) and max_penalty (a negative value, e.g., -0.1).

A helper function, `zipngram(text, ngram_size)`, generates n-grams by converting text to lowercase, splitting it into words, and using `zip(*[words[i:] for i in range(ngram_size)])` for efficient extraction.

Inside, `repetition_penalty_reward(...)` computes the penalty for each completion. If it's empty or too short, it gets a reward of 0.0.

The penalty scales as scaling `= 1 - len(ngrams) / total`, where total is the number of n-grams and len(ngrams) is the unique count. More repetition makes scaling approach 1, increasing the penalty.

The final reward is scaling * max_penalty, meaning less repetition results in a smaller penalty, while high repetition leads to a stronger negative reward. 

>We have implemented all five reward functions, Let’s move on to next stage where we define our training args

## Training Configurations for R1 Zero

Now we to code a configuration where we can fine-tune how our *reward functions* actually work. So, Let’s define that configuration class:

```python
# Define GRPOScriptArguments for reward function parameters
@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
```

Our `@dataclass` decorator makes it easy to create a class for storing data. WhileGRPOScriptArguments class holds reward settings.

The reward_funcs list decides which rewards to use, starting with ["accuracy", "format"], but you can add more like "reasoning_steps", "cosine", "repetition_penalty".

Some settings control how the cosine_scaled_reward and repetition_penalty_reward work, letting you adjust how rewards are given.

Next up, we have TrainingArguments from the transformers library. This is the **main** configuration object that controls almost **everything** about the training process.
```python
# Define TrainingArguments from transformers
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Output directory for checkpoints and logs
    overwrite_output_dir=True,
    num_train_epochs=1,             # Total number of training epochs
    per_device_train_batch_size=8,  # Batch size per device during training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
    learning_rate=5e-5,            # Initial learning rate for AdamW optimizer
    warmup_ratio=0.1,              # Linear warmup over warmup_ratio fraction of training steps
    weight_decay=0.01,             # Apply weight decay to all layers except bias and LayerNorm weights
    logging_steps=10,              # Log every X updates steps
    evaluation_strategy="steps",    # Evaluate every `eval_steps`
    eval_steps=50,                 # Evaluation and logging steps
    save_strategy="steps",         # Save checkpoint every `save_steps`
    save_steps=50,                 # Save checkpoint every X updates steps
    save_total_limit=2,            # Limit the total amount of checkpoints. Deletes the older checkpoints.
    dataloader_num_workers=2,      # Number of subprocesses to use for data loading
    seed=42,                       # Random seed for reproducibility
    bf16=True,                     # Use mixed precision BFP16 training
    push_to_hub=False,             # Whether to push the final model to Hugging Face Hub
    gradient_checkpointing=True,   # Enable gradient checkpointing
    report_to="none",              # Reporting to no one
)
```

Finally, we need to have a ModelConfig. This is where we put settings that are specific to the **model itself**, like which pre-trained model to use, what data type to use (like bfloat16), and whether to trust remote code or not and so.

Let’s define our ModelConfig:
```python
@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    model_name_or_path: str = field(
        default=MODEL_NAME, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )
```

Our **ModelConfig** class holds key settings, including model_name_or_path, which defaults to **Qwen 0.5B Instruct**. We use torch_dtype="bfloat16" for efficiency and set trust_remote_code=True for safe remote loading. Additionally, attn_implementation="flash_attention_2" is enabled for potentially faster training if supported.

Now we need to actually **create** instances of these configuration classes so we can use them:
```python
# Instantiate configuration objects
script_args = GRPOScriptArguments()
model_args = ModelConfig()
```

Next, we need to get our list of reward functions and any “callbacks” we want to use during training.

Callbacks are like little helpers that can do things at different points in the training process (like logging progress, saving models, etc.). For now, we’ll just use a simple logging callback.

Getting our reward functions in one place.
```python
# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of reward functions based on the script arguments.
    """
    reward_funcs_list = []
    reward_funcs_registry = {
        "accuracy": accuracy_reward,  # Assuming accuracy_reward is defined in previous steps
        "format": format_reward,      # Assuming format_reward is defined in previous steps
        "reasoning_steps": reasoning_steps_reward, # Assuming reasoning_steps_reward is defined
        "cosine": get_cosine_scaled_reward( # Assuming get_cosine_scaled_reward is defined
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward( # Assuming get_repetition_penalty_reward is defined
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
    }

    for func_name in script_args.reward_funcs:
        if func_name not in reward_funcs_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_funcs_list.append(reward_funcs_registry[func_name])

    return reward_funcs_list
```
Our callback function which will track loss and other important info.
```python
logger = logging.getLogger(__name__)

class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % args.logging_steps == 0:
            logger.info(f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}")

def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()] # Instantiate our LoggingCallback
    return callbacks
```

Finally, initializing these function.
```python
# Get reward functions and callbacks
reward_functions = get_reward_functions(script_args)
callbacks = get_callbacks(training_args, model_args, script_args)
```

## GRPO Training Loop

This is the engine that will actually drive our GRPO training. We need to initialize it, giving it all the pieces we’ve prepared: our model, reward functions, training arguments, dataset, and callbacks!

Let’s initialize the GRPOTrainer:
```python
# Create GRPOConfig from TrainingArguments
grpo_config = GRPOConfig(
    **training_args.to_dict(), # Convert TrainingArguments to dictionary and unpack
    **{ 
       # REMOVED model_init_kwargs here 
       # We are passing the instantiated 'model' object, so GRPOTrainer doesn't need model_init_kwargs
    }
)

grpo_trainer = GRPOTrainer(
    model=model,                      # Our initialized Qwen model
    reward_funcs=reward_functions,    # List of reward functions from previous step
    args=grpo_config,                # GRPOConfig (created from TrainingArguments)
    train_dataset=dataset['train'],   # Training dataset
    eval_dataset=dataset['test'],    # Evaluation dataset
    callbacks=callbacks              # List of callbacks
)
```

We can now start the **Training Loop**! This is as simple as calling the train() method on our grpo_trainer.
```python
# Start the GRPO Training Loop
train_result = grpo_trainer.train()
```
When you run this cell, you should see the training process begin.
```
...
INFO:__main__:Step 10: Loss = ..., Learning Rate = ...
INFO:__main__:Step 20: Loss = ..., Learning Rate = ...
...
```
Training will take some time but we set **num_train_epochs = 1** and are using a small model, it shouldn’t take *too* long for this example.

But for real-world GRPO DeepSeek R1 Zero training, you’d likely train for many more epochs and steps.

## Saving Tiny R1 Zero LLM

Once the training completed, we can save our trained model which can be used for inferencing.
```python
# Define the path to your trained model (same as OUTPUT_DIR)
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

# Save the trained model
grpo_trainer.save_model(TRAINED_MODEL_PATH)

print(f"GRPO Trained model saved to {TRAINED_MODEL_PATH}")
```
Then we can simply load the trained model using:
```python
# Load the tokenizer - make sure to use trust_remote_code=True if needed
tokenizer = AutoTokenizer.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model config requires it
    padding_side="right" # Ensure consistent padding side
)

# Set pad token if it wasn't saved or loaded correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the trained model itself
trained_model = AutoModelForCausalLM.from_pretrained(
    TRAINED_MODEL_PATH,
    trust_remote_code=True, # If your model architecture requires it
    torch_dtype=torch.bfloat16 # Keep the same dtype as training for consistency
)

# Move the loaded model to your device (GPU if available)
trained_model.to(device) # 'device' is still our CUDA device from before
```

In order to use it for inference:

```python
# Testing Inference with the Trained Model
def test_trained_model_inference(user_input: str):
    """Test inference with the loaded trained model and tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Re-use our system prompt
        {"role": "user", "content": user_input}
    ]

    # Apply chat template using our tokenizer
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate output using our *trained_model*
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=200, # Maybe generate a bit longer now
        do_sample=True,
        temperature=0.7
    )

    # Decode the generated tokens back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
```

## Two main problems with R1 Zero

Now that we have completed our R1 zero training approach using our base model Qwen2–0.5B instead of their DeepSeek V3 (original base model).

We cannot identify our trained model problems but researches of DeepSeek saw the R1 Zero model performed really well on reasoning tests, even scoring similarly to more advanced models like **OpenAI-01–0912** on tasks like **AIME 2024**.

This showed that using reinforcement learning (RL) to encourage reasoning in language models is a promising approach.

But they also noticed DeepSeek-R1-Zero had some key issues that needed fixing for real world use and wider research.

![Problem with R1 Zero (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6378/1*_NdVhpb9cgT3-8o3Qn7mMA.png)

Researchers of DeepSeek states that the template is *intentionally simple and structurally focused*. It *avoids* imposing any *content-specific* constraints on the *reasoning process itself*. For example, it doesn’t say:

* “You *must* use step-by-step reasoning” (It just says “reasoning process” leaving it open to the model to define what that means).

* “You *must* use reflective reasoning”

* “You *must* use a specific problem-solving strategy”

The main problem was that the reasoning processes inside the `<think>` tags were hard to read, making it tough for humans to follow and analyze.

Another issue was language mixing, when asked multi-lingual questions, the model sometimes mixed languages in the same response, leading to inconsistent and confusing outputs.

If you asked it questions in, say, Spanish. Suddenly, its “thinking” would be a jumbled mix of **English and Spanish, **not exactly polished! These problems, messy reasoning and language confusion, were the clear roadblocks.
> These are the two main reasons they transformed their initial R1 Zero Model into the R1

## Preparing Cold Start Data for SFT

So to fix R1 Zero issues and really get DeepSeek reasoning properly, researchers performed a **Cold Start Data Collection and included Supervised Fine Tuning**.

You can think of it as giving the model a good foundation in reasoning before the really intense RL training. Basically, they wanted to teach **DeepSeek-V3 Base** what good reasoning looks like and how to present it clearly.

One of the example of cold start data is [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) that we see earlier and will be using for creating R1, but **we need to understand how cold dataset is created so we wont skip any part from the actual training**.

## Few-shot Prompting with Long CoT

One technique is **Few-shot Prompting with Long Chain-of-Thought (CoT),** in which we try to show DeepSeek-V3 Base (or in our case, Qwen2–0.5B) few examples of questions paired with super detailed, step-by-step solutions. This is Chain-of-Thought (CoT).

![Long CoT (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4068/1*SAhvB0JqaK4d45IiIcj1Ow.png)

Goal of this approach is to make the model learn by example and start mimicking this thorough reasoning style.

For our example problem “What is 2 + 3 * 4?”, we can create prompts that include a few solved problems as examples. Let’s see how this looks in Python:
```python
# Loading Model and Tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate Long COT Response
def generate_response(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step solutions."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False) # Keep it deterministic for example
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|im_start|>assistant\n")[-1].strip() # Extract assistant's response
```
Let’s define the few shot examples accordingly for our asked question:
```python
# Example problems with solutions (using | special_token | as delimiter)
few_shot_prompt = """
Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution:
"""
```

Now using our base model our sample generations looks like this:
```python
# Generate response for the target problem using few-shot examples
target_problem_prompt = few_shot_prompt + "What is 2 + 3 * 4?"
model_response_few_shot = generate_response(target_problem_prompt)

print("Few-shot Prompt:")
print(target_problem_prompt)
print("\nModel Response (Few-shot CoT):")
print(model_response_few_shot)
```

It output this structured data

```
Few-shot Prompt:
Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, 
which is 3. Then, add 5 to 3.  3 + 5 equals 8. 
<|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. 
Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. 
<|special_token|> Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution: 

Model Response (Few-shot CoT):
<|special_token|> To solve 2 + 3 * 4, we need to follow the order 
of operations (PEMDAS/BODMAS). Multiplication should be performed 
before addition.
Step 1: Multiply 3 by 4, which equals 12.
Step 2: Add 2 to the result from Step 1: 2 + 12 = 14.
<|special_token|> Summary: The answer is 14.
```

See how the model, after seeing examples, starts to structure its answer with <|special_token|> delimiters and provides step-by-step reasoning leading to the summary and final answer!

This is the power of few-shot learning guiding the model towards the desired output format.

## Direct Prompting

Another method is **Direct Prompting**. Here, we directly instruct the model to not just solve the problem, but also to explicitly show its reasoning step-by-step and then verify its answer. 

This is about encouraging a more deliberate and thoughtful problem-solving approach.

![Example based learning (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4656/1*IYyk7UWgDNADFe_djWcXow.png)

Let’s craft a prompt for “What is 2 + 3 * 4?” that explicitly asks for reasoning and verification. Here’s the Python code to see it in action:
```python
# Direct prompting example
direct_prompt_text = """
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?
"""

model_response_direct = generate_response(direct_prompt_text)

print("Direct Prompt:")
print(direct_prompt_text)
print("\nModel Response (Direct Prompting):")
print(model_response_direct)
```
The direct prompting output is very easy to understand and this is what it looks like:
```
Direct Prompt:
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?

Model Response (Direct Prompting):
<|special_token|> Reasoning: To solve 2 + 3 * 4, I need to follow 
the order of operations, which states that multiplication should 
be done before addition.
Step 1: Multiply 3 by 4, which equals 12.
Step 2: Add 2 to the result from Step 1: 2 + 12 = 14.
Verification: To verify the answer, I can double-check the 
order of operations and the calculations. Multiplication is 
indeed performed before addition, and the calculations are correct.
<|special_token|> Summary: The answer is 14.
```
As you can see, by directly asking for reasoning and verification, the model provides a more comprehensive output, including a “Verification” section.

This method directly guides the model to produce the kind of detailed reasoning we are looking for.

## Post Processing Refinement

The final technique involves **Post-Processing Refinement**. Interestingly, they even used the outputs from the already trained R1 Zero model for this!

Even with its issues, R1 Zero could reason somewhat. So, they took R1 Zero outputs and had human annotators refine them, making them cleaner, more structured, and correcting any mistakes.

![Processing Refnement (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/4388/1*-GR29EAnTOVBarQ2JrF5sA.png)

Imagine a messy R1 Zero output like this:
```
<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>
<answer> 14 </answer>
```

Human annotators would then refine it to something much clearer and better formatted:
```
<|special_token|> Reasoning: To solve this, we use order of operations, doing multiplication before addition.
Step 1: Multiply 3 by 4, which is 12.
Step 2: Add 2 to the result: 2 + 12 = 14.
<|special_token|> Summary: The answer is 14.
```

While we can’t perfectly simulate human refinement in code, we can demonstrate a basic idea of how you might programmatically reformat and structure a potentially messy output. 

Let’s take a simulated “messy” output and show how we could refine it:
```python
# Simulated messy R1 Zero output
messy_output = "<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>\n<answer> 14 </answer>"

def refine_output(messy_text):
    think_content = messy_text.split("<think>")[1].split("</think>")[0].strip()
    answer_content = messy_text.split("<answer>")[1].split("</answer>")[0].strip()

    refined_text = f"""<|special_token|> Reasoning: {think_content.replace('umm...', '').strip().capitalize()}.
<|special_token|> Summary: The answer is {answer_content}."""
    return refined_text

refined_output_text = refine_output(messy_output)

print("Messy Output (Simulated R1 Zero):")
print(messy_output)
print("\nRefined Output:")
print(refined_output_text)
```

This will output:
```
Messy Output (Simulated R1 Zero):
<think>  ummm... multiply 3 and 4... get 12... then add 2...</think>
<answer> 14 </answer>

Refined Output:
<|special_token|> Reasoning: Multiply 3 and 4... get 12... then add 2.
<|special_token|> Summary: The answer is 14.
```

This simple refine_output function is just a basic example. Real refinement by humans involves much more nuanced understanding and correction of reasoning steps.

However, it shows the core idea: taking initial model outputs and improving their quality and structure to create better training data.
> After generating this Cold Start Data, the next crucial step was **Supervised Fine-Tuning (SFT)**, which we’ll explore in the next section!

## SFT Stage 1 With Cold Start Data

To generate proper cold start data to build R1 using Supervised fine-tuning, we obviously need a proper team along with an excessive amount of code, but thankfully, we already have data ([Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)) that is similar to the cold start form.
> We need to know what and how training happens inside the SFT Trainer as it processes our training data?

SFT is a form of supervised learning. This means we’re giving the model pairs of inputs and *desired* outputs.

In our case, the input might be a problem prompt, and the desired output is the well-reasoned, step-by-step solution from our training dataset. **I hope this point gives a clear view of why there is a need of cold data.**

It takes our tokenized training data and feeds it to the model in batches. For each batch, a important set of operations happens, Let’s visualize this internal process:

![SFT WorkFlow (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6838/1*EsEgATw1aSYPjfGtpId2mQ.png)

First, the model takes an input, a problem prompt, for instance. It processes this input and generates its best guess for the solution, token by token. These are the *predicted tokens*.

Next, the SFT Trainer needs to know how good (or bad) these predictions are. It uses a *loss function*, typically Cross-Entropy Loss. This function mathematically compares the model’s predicted tokens to the *correct* tokens from our training data. Think of it as calculating the “error” of the model’s answer.

This “error” isn’t just discarded. It’s the crucial signal for learning. Through a process called *backpropagation*, this error is used to calculate *gradients*. Gradients are like guides, pointing in the direction of parameter adjustments that would reduce the error.

Finally, an *optimizer*, like **AdamW** uses these gradients to subtly tweak the model’s internal settings — its parameters. These tweaks are designed to make the model’s next prediction a little bit closer to the correct answer.

## Stage 1 SFT Trainer Configs for R1

Remember those problems we had with R1 Zero messy reasoning and language mixing? SFT is designed to fix exactly that. By training on high-quality, refined data, we’re teaching the model:

* **Clear Reasoning Style**: To structure its “thinking” in a way that’s easy to read and follow.

* **Consistent Language**: To stick to one language within a response, avoiding confusing mixes.

We’re using the Bespoke-Stratos-17k dataset for SFT. As we saw earlier, it’s got 17,000 problems focused on math and code, with a format that looks pretty good for our needs.

Let’s quickly remind ourselves of a sample from Bespoke-Stratos-17k:
```python
# Load the "Bespoke-Stratos-17k" dataset from bespokelabs
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# Access the first sample in the training set
bespoke_rl['train'][0]


#### OUTPUT ####
{
  'system': 'Your role as an assistant involves ... ',
  'conversations': [{'from': 'user', 'value': 'Return your ...'}]
}
#### OUTPUT ####
```

This dataset, with its system prompts and user-assistant conversations, is perfect for showing our model how conversations with reasoning should look.

We’ll use the trl library again, which makes SFT training super easy.

First, we need to set up our configurations, similar to what we did for GRPO, but this time for SFT.
```python
# Model and Output Configuration (same as before, or adjust as needed)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "data/Qwen-SFT-training" # New output directory for SFT model
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Arguments - similar to GRPO, but adjust for SFT
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,         # Adjust epochs as needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,        # Adjust learning rate for SFT
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="no",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    dataloader_num_workers=2,
    seed=42,
    bf16=True,
    push_to_hub=False,
    gradient_checkpointing=True,
    report_to="none",
    packing=True, # Enable data packing for efficiency
    max_seq_length=4096 # Set max sequence length
)

# Model Configuration - same as before
model_args = ModelConfig(
    model_name_or_path=MODEL_NAME,
    model_revision="main",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
```

These TrainingArguments and ModelConfig are quite similar to what we used for GRPO, but with a few tweaks that are more suitable for SFT (like a slightly different learning rate, and importantly, packing=True and max_seq_length=4096 for efficient training on longer sequences).

## Stage 1 STF Training Loop

Now, let’s load our dataset and tokenizer:
```python
# Load Bespoke-Stratos-17k dataset
dataset_sft = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split='train') # Only using train split for simplicity

# Initialize tokenizer - same as before
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

And finally, we initialize the SFTTrainer and start training!
```python
# Initialize base model for SFT - same as before
model_sft = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Initialize the SFT Trainer
sft_trainer = SFTTrainer(
    model=model_sft,                     # Our initialized Qwen model
    train_dataset=dataset_sft,           # Bespoke-Stratos-17k dataset
    tokenizer=tokenizer,                 # Tokenizer
    args=training_args,                  # Training arguments
    dataset_text_field="conversations",  # Field in dataset containing text - IMPORTANT for SFT
    packing=True,                        # Enable data packing
    max_seq_length=4096                 # Max sequence length
)

# Start the SFT Training Loop
sft_train_result = sft_trainer.train()
```

When you run this code, you’ll see the SFT training process start. It will look similar to the GRPO training output, showing loss and learning rate at each logging step.
```
...
INFO:__main__:Step 10: Loss = ..., Learning Rate = ...
INFO:__main__:Step 20: Loss = ..., Learning Rate = ...
...
```

Just like with GRPO, training time will depend on your hardware and chosen epochs. Since we’re still using a small model and only 1 epoch for this example, it should be reasonably quick.

## Saving Tiny R1 LLM

After SFT is done, we save our newly fine-tuned model (R1).
```python
# Saving the Trained SFT Model
TRAINED_SFT_MODEL_PATH = "data/Qwen-SFT-training" # Same as OUTPUT_DIR

# Save the tokenizer
tokenizer.save_pretrained(TRAINED_SFT_MODEL_PATH)

# Save the trained model
sft_trainer.save_model(TRAINED_SFT_MODEL_PATH)

print(f"SFT Trained model saved to {TRAINED_SFT_MODEL_PATH}")
```

And that’s it for the SFT part! We’ve now taken our base model, shown it lots of examples of good reasoning, and fine-tuned it to be better at producing clear, structured responses.
> This finetuned model using SFT is what we called R1 after SFT stage 1

The steps after SFT, especially the RL stages and rejection sampling, are complex to implement from scratch in Python. Focusing on the theoretical understanding is key to understand the overall process.

## Reasoning-Oriented RL

After SFT, the model can reason better, but we want to *really* focus on reasoning quality and fix language mixing. This stage uses RL again, but with a smarter reward system.

This new reward checks if the model reasoning and answer are in the same language as the question. If you ask in English, the *whole* response should be in English. This fixes language mixing issues.

![Reasoning Oriented RL (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7468/1*Z2oHDdkWb7RnO5uVHPSvMg.png)

It adds a **Language Consistency Reward** alongside accuracy to ensure the SFT model reasons and answers in the same language as the input.

The GRPO algorithm and training loop from R1 Zero are reused, but the reward signals are improved to specifically target better reasoning and consistent language output.

## Rejection Sampling

To get super high-quality reasoning data, DeepSeek uses **Rejection Sampling**. Think of it as a filter to keep only the *best* examples.

![Rejection Sampling (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/8520/1*obG-BrhwtIuOv7YBZIpSwg.png)

The model generates many reasoning examples. These are then evaluated for correctness and reasoning quality (often using a generative reward model and human checks).

Only the *best*, high-quality reasoning examples are kept. Combined with non-reasoning data, this refined dataset is used for a second **SFT Stage 2**, further improving reasoning and general abilities.

## SFT Stage 2 Training

The final RL stage focuses on making the model a helpful and safe AI assistant for *all* situations, not just reasoning problems. This is about alignment with human values.

**Key Focus: Helpfulness & Harmlessness Rewards**

Not just accuracy, the reward system now includes:

* **Helpfulness:** Is the response useful and informative?

* **Harmlessness:** Is the response safe, unbiased, and ethical?

![SFT Stage 2 (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/7086/1*_u5ALx4VYQpsSgT_0s10HQ.png)

The training data becomes diverse, including reasoning tasks and human preference data (which output is better — more helpful, less harmful?).

The reward system now balances accuracy with **helpfulness and harmlessness**. Iterative RL training (likely GRPO again) optimizes the model to be not just good at reasoning, but also a safe and helpful AI assistant for general use, resulting in DeepSeek R1.

## Distillation

To make DeepSeek R1 accessible, they **distilled** its knowledge into smaller models.

![Distillation Process (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/2500/0*QdOxtvuKaEASreK0.png)

Distillation takes the knowledge of a large, powerful “teacher” model (DeepSeek R1) and transfers it to smaller “student” models. Using a large dataset of reasoning examples, the outputs of DeepSeek R1 are used as the *target* answers.

Smaller models are then trained (SFT) to mimic these outputs. This results in smaller, faster models that retain a significant portion of DeepSeek R1’s reasoning abilities, making them more practical for wider use.

Happy reading!

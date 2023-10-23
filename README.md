
#MiniGPT: Building GPT2 from scratch
MiniGPT is a lightweight implementation of the Generative Pre-trained Transformer (GPT) model for natural language processing tasks. It is designed to be simple and easy to use, making it a great option for small-scale applications or for learning and experimenting with generative models.



## Installation
1. Clone the repo and run `pip install -r requirements.txt`

## Usage
1. Run `tokenizer/train_tokenizer.py` to generate the tokenizer file. The model will tokenize text based on it.
2. Run `datasets/prepare_dataset.py` to generate dataset files.
3. Run `train.py` to start training~

**Modify the files stated above if you wish to change their parameters.**

## Usage (Inference)
To edit model generation parameters, head over `inference.py` to this section:
```py
# Parameters (Edit here):
n_tokens = 1000
temperature = 0.8
top_k = 0
top_p = 0.9
model_path = 'models/smallGPT.pth'

```



## smallGPT
## Advantages of smallGPT:

1. **From Scratch Efficiency:**smallGPT is developed from the ground up, offering a streamlined approach to the esteemed GPT model. It showcases remarkable efficiency while maintaining a slight trade-off in quality.

2. **Learning Playground:** Designed for individuals eager to delve into the world of AI,smallGPT's architecture offers a unique opportunity to grasp the inner workings of generative models. It's a launchpad for honing your skills and deepening your understanding.

3. **Small-Scale Powerhouse:** Beyond learning and experimentation,smallGPT is a suitable option for small-scale applications. It empowers you to integrate AI-powered language generation into projects where efficiency and performance are paramount.

## Empowerment Through Adaptability:

1. **Customization Capabilities:** smallGPT's adaptability empowers you to modify and fine-tune the model to suit your specific goals, offering a canvas for creating AI solutions tailored to your requirements.

2. **Learning Journey:** Use smallGPT as a stepping stone to comprehend the foundations of generative models. Its accessible design and documentation provide an ideal environment for those new to AI.

3. **Experimentation Lab:** Engage in experiments by tweaking and testing microGPT's parameters. The model's simplicity and versatility provide a fertile ground for innovation.

## Contribution
If you would like to contribute, please follow these guidelines:




## Resources
- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Let's build GPT from scratch](https://youtu.be/kCc8FmEb1nY)
- [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/pdf/2304.08442.pdf)
- [Huggingface GPT 2 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt2)





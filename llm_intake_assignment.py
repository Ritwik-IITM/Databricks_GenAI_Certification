# Databricks notebook source
# MAGIC %md
# MAGIC # <font color="#76b900">**2:** LLM Architecture Intuitions</font>

# COMMAND ----------

# MAGIC %md
# MAGIC In the last notebook, you touched the surface-level interface of the HuggingFace &#x1F917; pipelines and went a single layer deeper, seeing the abstractions associated with the pipeline and looking a little under the hood to how these components might be implemented. Namely, you should now be familiar with the `preprocess -> forward -> postprocess` abstraction which hides the complexity from the user and makes it easy to work with your models. In this notebook, we'll be looking a bit deeper to try and understand what techniques are being used to facilitate this reasoning.
# MAGIC
# MAGIC #### **Learning Objectives:**
# MAGIC
# MAGIC - Tokenization and embedding intuitions, specifically relating to how the data comes into our models and what properties the network can take advantage of.
# MAGIC - Transformer encoder architectures for performing sequence-level reasoning for an n-sequence-in-n-sequence-out formulation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1. Getting The Model Inputs
# MAGIC As we saw previously, the overall pipeline has to convert to and from the tensor representation using the `preprocess` and `postprocess` functionalities. Looking a little deeper, we can see that the preprocess method relies on the tokenizer, and we can assume that the postprocess does too, so let's look at that pipeline again:

# COMMAND ----------

from transformers import BertTokenizer, BertModel, FillMaskPipeline, AutoModelForMaskedLM

class MyMlmPipeline(FillMaskPipeline):
    def __init__(self):
        super().__init__(
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased'),
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        )

    def __call__(self, string, verbose=False):
        ## Verbose argument just there for our convenience
        input_tensors = self.preprocess(string)
        output_tensors = self.forward(input_tensors)
        output = self.postprocess(output_tensors)
        return output

    def preprocess(self, string):
        string = [string] if isinstance(string, str) else string
        inputs = self.tokenizer(string, return_tensors="pt")
        return inputs

    def forward(self, tensor_dict):
        output_tensors = self.model.forward(**tensor_dict)
        return {**output_tensors, **tensor_dict}

    def postprocess(self, tensor_dict):
        ## Very Task-specific; see FillMaskPipeline.postprocess
        return super().postprocess(tensor_dict)


class MyMlmSubPipeline(MyMlmPipeline):
    def __call__(self, string, verbose=False):
        ## Verbose argument just there for our convenience
        input_tensors = self.preprocess(string)
        return input_tensors
        # output_tensors = self.forward(input_tensors)
        # output = self.postprocess(output_tensors)
        # return output


unmasker = MyMlmSubPipeline()
unmasker("Hello, Mr. Bert! How is it [MASK]?", verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC This shows that the tokenizer is a conversion strategy for converting the input string to a series of tokens. A token is a symbolic representation of something, and is usually reasoned about as a class. Within the scope of language modeling, a token is usually a word, or a letter, or some other substring that can be used as a fundamental building block of a sentence. This is one of the more consistent things among all of the large language models you'll encounter, and also probably one of the conceptually-simplest. Still, it's nice to know what they are and how they operate.

# COMMAND ----------

# MAGIC %md
# MAGIC When given our string, the tokenizer responds with several components:
# MAGIC - `input_ids`: These are just the IDs of the tokens that make up our sentence. Said tokens can be words, punctuation, letters, whatever. Just individual entries out of a set vocabulary, exactly like classes.
# MAGIC     - Try the following:
# MAGIC     ```python
# MAGIC     msg = "Hello world and have a great day!"
# MAGIC     unmasker.tokenizer.tokenize(msg)       ## See token boundaries
# MAGIC     # x = unmasker.tokenizer.encode(msg)   ## See special tokens at end
# MAGIC     # x = unmasker.tokenizer.decode(x)     ## See decoding
# MAGIC     # print(x)
# MAGIC     ```
# MAGIC
# MAGIC - `token_type_ids`: Added information that the BERT authors realized was useful. This is just an extra flag which tells BERT whether this is the first or the second sentence. This can be useful sometimes (and is a major part of the training objective of BERT specifically), but you'll probably never use it knowingly in practice.
# MAGIC     - Try the following:
# MAGIC     ```python
# MAGIC     unmasker.tokenizer("Hello world!", "Have a great day!")
# MAGIC     ```
# MAGIC
# MAGIC - `attention_mask`: To be discussed later; It's a natural input to transformer components and regulates what tokens a specific token can pay attention to. For BERT, this is not necessary, but can be specified.
# MAGIC
# MAGIC As far as we will need to be concerned, the `input_ids` are the most important input segment for our model. Considering this, we can intuit how the LLMs approach the task of natural language processing; as reasoning about an **ordered sequence of tokens.** On one hand this should be somewhat reassuring, as classification is a common task in deep learning that you're probably well-familiar with. On the other, you may be a little less familiar with either the process of taking in classes as input or reasoning about sequences. We can go ahead and investigate the model to try to figure out what intuitions the language models might be using to make these problems tractible. 

# COMMAND ----------

msg = "Hello world and have a great day!"
unmasker.tokenizer.tokenize(msg)       ## See token boundaries
# x = unmasker.tokenizer.encode(msg)   ## See special tokens at end
# x = unmasker.tokenizer.decode(x)     ## See decoding
# print(x)

# COMMAND ----------

unmasker.tokenizer.encode(msg)

# COMMAND ----------

unmasker.tokenizer.decode(unmasker.tokenizer.encode(msg))

# COMMAND ----------

unmasker.tokenizer("Hello world!", "Have a great day!")

# COMMAND ----------

unmasker.tokenizer.decode([101, 7592, 2088, 999, 102, 2031, 1037, 2307, 2154, 999, 102])

# COMMAND ----------

## Feel free to run some code cells here -> encoding and decoding!

# ASSESSMENT -> needs to be submitted before 14th April

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2. Capturing Token Semantics
# MAGIC
# MAGIC We now know that natural language reasoning is a problem of inferring insights from an ordered sequence of tokens, so how would we approach that formulation? With regards to classes, we should already have some intuitions:
# MAGIC
# MAGIC - **On the output side**, we can output a probability distribution over the set of possible classes. For example, if we were predicting among `cat`, `dog`, and `bird`, we can output a 3-value vector with the intuitive meaning of `<is_cat, is_dog, is_bird>`. For ground-truth, you just use one-hot encodings where the realized instance is 1 and the other options are 0.
# MAGIC - **On the input side**, we could also feed in a one-hot value if we want, but a more efficient strategy when you're primarily dealing with one-hots is to use an **Embedding Layer**, or a glorified matrix where the class index is the row to access. Whichever one you choose, you'll be keeping a record of the semantics associated with the class in your model architecture (either in the weights of the first layer or the weights of the lookup matrix).
# MAGIC
# MAGIC With that said, the LLM definitely has a strategy for this:

# COMMAND ----------

model = unmasker.model
dir(model)
#dir(model.bert)
#print(model.bert.embeddings)
# print(model.bert.embeddings.word_embeddings)
# print(model.bert.embeddings.position_embeddings)
# print(model.bert.embeddings.token_type_embeddings)


# 10 X 768

# 10 X 768

# 2 X 768



# COMMAND ----------

print(model.bert.embeddings.word_embeddings)
print(model.bert.embeddings.position_embeddings)
print(model.bert.embeddings.token_type_embeddings)

# COMMAND ----------

print(model.bert.embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC From this, we can identify the 3 components discussed in the presentation:
# MAGIC - **Word Embeddings:** Semantic vectors representing the input tokens.
# MAGIC - **Position Embeddings**: Semantic vectors representing the position of the words.
# MAGIC - **Token Type Embedding**: Semantic vectors representing whether the token belongs to the first or second sentence.
# MAGIC
# MAGIC Notice how the `Embedding` component is constructed with the format:
# MAGIC
# MAGIC ```
# MAGIC Embedding(in_channel, out_channel)
# MAGIC ```
# MAGIC
# MAGIC We can see from this that BERT uses 768-dimensional embeddings, and can speculate on how they are obtained. The word embeddings seem to be coming from a 30,522-dimensional vector (the number of unique tokens in the vocabulary), the positional ones from 512, and the token types from just a few. Let's explore these a bit further.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Investigating the Word Embeddings
# MAGIC
# MAGIC Let's go ahead and take a look at the word embeddings:

# COMMAND ----------

import torch

tokenizer = unmasker.tokenizer

def get_word_embeddings(string):
    tokens = tokenizer(string)['input_ids']
    tokens = tokens[1:-1] ## Remove cls and sep tokens
    tokens = torch.tensor(tokens)
    return model.bert.embeddings.word_embeddings(tokens)

## Pre-spaced to show where the tokens are. Same results without extra spaces
string = "Hello World From Me, my cat and my dog!"
tokens = [tokenizer.convert_ids_to_tokens(x) for x in tokenizer.encode(string)[1:-1]]
embeddings = get_word_embeddings(string)
print(embeddings.shape)
embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Given what we talked about with embedding vectors, we'd expect the word embedding vectors to capture some of the meanings of the tokens that span our intended natural language. To investigate it, we can go ahead and define some helper functions:

# COMMAND ----------

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def cosine_similarity(x1, x2):
    """Compute cosine similarity between two vectors."""
    dot_product = x1 @ x2.T
    norm_x1 = torch.norm(x1, dim=-1)
    norm_x2 = torch.norm(x2, dim=-1)
    return dot_product / (norm_x1 * norm_x2)

def scaled_dp_similarity(x1, x2):
    """Compute dot-product similarity between two vectors."""
    dot_product = x1 @ x2.T
    d = torch.sqrt(torch.tensor(x1.shape[-1]))
    return dot_product / d

def softmax_similarity(x1, x2):
    """Compute softmaxed dp similarity between two vectors."""
    out = scaled_dp_similarity(x1, x2)
    return torch.softmax(out, dim=1)

def plot_mtx(matrix, name='', tokens=[]):
    """Compute similarity matrix for embeddings."""
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    label_dict = {} if tokens is None else {'xticklabels' : tokens, 'yticklabels': tokens}
    sns.heatmap(
        np.round(matrix.detach().numpy(), 3),
        annot=True, cmap='coolwarm',
        # vmin=-1, vmax=1,
        **label_dict
    )
    plt.title(f"Embedding {name} Matrix")
    plt.yticks(rotation=0)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC With these visualization and metric functions defined, we can view the similarities of the embeddings in different measurement spaces:
# MAGIC
# MAGIC - The following will compute the cosine similarity:
# MAGIC   ```python
# MAGIC   plot_mtx(cosine_similarity(embeddings, embeddings), 'Cosine Sim', tokens)
# MAGIC   ```
# MAGIC   You'll notice that we do get some pretty nice properties, and the result is a nice normalized matrix, but unfortunately this throws away distance-related information of the vectors.
# MAGIC
# MAGIC - As we'll soon see this idea being incorporated into the architecture, it's worth investigating what happens when we decide to transition to softmax-based similarity:
# MAGIC   ```python
# MAGIC   plot_mtx(softmax_similarity(embeddings, embeddings), 'Softmax(x1) Sim', tokens)
# MAGIC   ```
# MAGIC   You'll see that the matrix is no longer symetric since we're applying softmax on a per-row basis, but it does have a nice intuitive analogue when you format it as a matrix product:
# MAGIC   **Relative to the others, how much does a token contribute to every other token?** This formulation will come up later as "attention."
# MAGIC
# MAGIC   You'll also notice that the magnitudes are pretty small, but we can increase the magnitude of the embeddings and observe a much more polarizing similarity matrix.
# MAGIC   ```python
# MAGIC   plot_mtx(softmax_similarity(embeddings*10, embeddings*10), 'Softmax(x10) Sim', tokens)
# MAGIC   ```
# MAGIC   
# MAGIC   Because the metric now factors magnitude into the decision process but keeps the output bounded and under control, this is a great choice when you actually want to inject similarity into optimization (again, foreshadowing).
# MAGIC
# MAGIC Regardless, the takehome message for word embeddings is roughly **"learned vector representation for each token based on its meaning and usage in sentences."**

# COMMAND ----------

## Please run the code lines and observe what happens

#plot_mtx(cosine_similarity(embeddings, embeddings), 'Cosine Sim', tokens)
#plot_mtx(scaled_dp_similarity(embeddings, embeddings), 'Dot Product Sim', tokens)
plot_mtx(softmax_similarity(embeddings, embeddings), 'Softmax/Prob Sim', tokens)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Investigating the Positional Embeddings
# MAGIC
# MAGIC Now that we've seen the word embeddings, we can take a look at the positional embeddings:
# MAGIC
# MAGIC ```python
# MAGIC model.bert.embeddings.position_embeddings ## -> Embedding(512, 768)
# MAGIC ```
# MAGIC
# MAGIC In contrast to the word embeddings, there is a new input dimension: 512.
# MAGIC
# MAGIC This actually corresponds to the number of input tokens that the BERT model can take in. All modern language models have a limited amount of tokens that can be fed in as a single input entry, and so there are only 512 possible positions to account for for our model.  
# MAGIC - **NOTE:** This limit is actually not a hard limit, and is implemented on a per-model basis due to steep performance degredation. More on this when we talk about attention.

# COMMAND ----------

def get_pos_embeddings(string):
    ## NOTE: In the previous method, we removed surrounding tokens for illustration only.
    ## For this one, we will not do the same since the index offset matters.
    tokens = tokenizer(string)['input_ids']  
    return model.bert.embeddings.position_embeddings(torch.arange(len(tokens)))

## Pre-spaced to show where the tokens are. Same results without extra spaces
string = "Hello World From Me, my cat and my dog!"
pos_embeddings = get_pos_embeddings(string)
print(pos_embeddings.shape)
pos_embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC The main difference you may have noticed is that instead of feeding in the tokens directly into the embedding layers, we're only feeding in a sequence of indices, literally via `torch.arange(n) = torch.tensor([0, 1, ..., n-1])`. The original "Transformers Is All You Need" paper used Positional "Encoding" which are pre-computed by a sinosoidal algorithm, but we can see that BERT just directly optimizes them instead. If it works, it works!
# MAGIC
# MAGIC You'll notice that the positional embedding has a more predictable and uniform cosine similarity plots compared to the word embeddings, which are all actually pretty consistent with a few key exceptions.
# MAGIC
# MAGIC ```python
# MAGIC plot_mtx(cosine_similarity(pos_embeddings, pos_embeddings), 'Cosine Sim', tokens)
# MAGIC ```
# MAGIC
# MAGIC **You're free to visualize a subset of the positional embeddings matrix below.**

# COMMAND ----------

plot_mtx(cosine_similarity(pos_embeddings, pos_embeddings), 'Cosine Sim', tokens)

# COMMAND ----------

## Please run the code lines and observe what happens

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Tail-End of the Embedding
# MAGIC
# MAGIC To wrap up our embedding discussions, we do still have our **token_type_embedding** embeddings, but they follow roughly the same logic; just take in some extra semantic information about the sentence structure, and encode it in. The authors saw that this extra bit of information was necessary, so the overall embedding definition for BERT is:
# MAGIC
# MAGIC `embed = WordEmbed[token] + PosEmbed[pos] + TypeEmbed[pos]`

# COMMAND ----------

model.bert.embeddings

# COMMAND ----------

# MAGIC %md
# MAGIC Then at the end, the LayerNorm section and Dropout section are also included, and these will permiate your architectures going forward. A light discussion is sufficient to motivate them:
# MAGIC
# MAGIC - The [LayerNorm layer](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) normalizes the data flowing through it so that each minibatch subscribes to a similar distribution. You've probably seen [BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) from computer vision; this has a similar logic, but now the normalization covers the layer outputs instead of the batch.
# MAGIC     - [Build Better Deep Learning Models with Batch and Layer Normalization | **PineCone.io**](https://www.pinecone.io/learn/batch-layer-normalization/)
# MAGIC     - [**PowerNorm** paper](https://arxiv.org/abs/2003.07845): Contains a deeper analyzes of Batch/Layer Norm and problems for the LLM use-case
# MAGIC - The [Dropout layer](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) just masks out some of the values during training. You've probably seen this before, and the logic is the same as usual; prevent over-reliance on a selection of features and distribute the logic throughout the network.
# MAGIC
# MAGIC From here, it is useful to remind you that HuggingFace is an open-source platform! Though it is quite large, its logic becomes transparent when you know where to look. In this case, we can see the code for how all of these things come together in [`transformers/models/bert/modeling_bert.py`](https://github.com/huggingface/transformers/blob/0a365c3e6a0e174302debff4023182838607acf1/src/transformers/models/bert/modeling_bert.py#L180C11-L180C11). Perusing the source code can help answer ambiguities about technical details such as "is this addition or concatenation" (it's addition) or "are there additional steps necessary to make this scheme work in practice" (yes). Please check it out and try to appreciate how litte information is actually necessary for the model to perform its reasoning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3. From Token-Level Reasoning to Passage-Level Reasoning
# MAGIC
# MAGIC **To summarize the key points of the LLM intake strategy:**
# MAGIC
# MAGIC - We take in a passage as an ordered sequence of tokens which we obtain by passing the string through the tokenizer.
# MAGIC - We train up embeddings corresponding to the token features (meanings, positions, etc.), and incorporate them together (in this case, literally by adding them).  
# MAGIC
# MAGIC **With this, we have some obvious options for how to reason about our data:**
# MAGIC
# MAGIC - We can just take our sequence of tokens, and then reason about each one of those one-at-a-time. This is quite similar to what we did in classification tasks, so we know it does work. 
# MAGIC     - **Problem:** This isn't good enough for text passages since the tokens have to reason about the other tokens in the sequence.
# MAGIC - On the other hand, we can try to reason about these things all at once by combining them and passing the data through dense layers
# MAGIC     - **Problem:** This will create a dense neural network that is intractable to optimize.
# MAGIC
# MAGIC The LLM solution is to do something between those two options: Allow reasoning to be done on each token, but also allow for small opportunities in which the network can combine the token reasoning and consider the sequence as a whole! That's where the **transformer** components come in!
# MAGIC
# MAGIC ### Transformer Attention
# MAGIC
# MAGIC **Transformers** are deep learning components described in the 2017 paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) for reasoning with language tasks, and the resulting architecture has been making its way into almost every state-of-the-art language modeling architecture since. This architecture uses an ***attention mechanism*** to create an interface where the other entries of the sequence can communicate semantic information to other tokens in the series.
# MAGIC
# MAGIC The formulation goes as follows: If we have semantic and positional information present in our embeddings, we can train a mapping from our embeddings into three semantic spaces $K$, $Q$, and $V$:
# MAGIC
# MAGIC - `Key` and `Query` are arguments to a similarity function (recall scaled softmax attention) to guage how much weight (or attention) should be given between any pair of sequence entries in the input.
# MAGIC     - In practice, the inputs to a specific transformer block are latent embeddings of the original tokens.
# MAGIC - `Value` is the information that should pass through to the next component, and is weighted by `SoftmaxAttention(Key, Query)` to produce an output that is positionally and semantically motivated.
# MAGIC
# MAGIC **In other words:** Given a semantic/position-rich sequence of $d_k$-element embeddings ($S$) and three dense layers ($K$, $Q$, and $V$) that operate per-sequence-entry, we can train a neural network to make semantic/position-driven predictions via the forward pass equation:
# MAGIC
# MAGIC $$\text{Self-Attention} = \text{SoftmaxAttention}(K_s, Q_s) \cdot V_s$$$$= \frac{K_s Q_s ^T}{\sqrt{d_k}}V_s$$
# MAGIC
# MAGIC <div><img src="imgs/attention-logic.png" width="1000"/></div>
# MAGIC
# MAGIC **Key Observations:**
# MAGIC
# MAGIC - Since the embeddings have both semantic and positional information, this will be able to reason about both the general meanings and the word order of an input sequence.
# MAGIC - Since scaled softmax attention is being used, the magnitude and cosine similarity of `Key` and `Query` all play a role in the decision-making process while the optimized result remains nicely-bounded.
# MAGIC - Since sequence length dimension gets preserved by matrix multiplication, there is a nice interpretation to the resulting attention matrix: **"What percentage of attention should each token pay to its surrounding tokens"**.
# MAGIC
# MAGIC This flavor of attention is called **self-attention**, since all of the `Key`, `Query`, and `Value` vectors are inferred from the same sequence. Other flavors will be presented later as necessary.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seeing Attention in the BERT Encoder
# MAGIC
# MAGIC Now that we've reviewed the logic of how self-attention works, let's look through the BERT encoder to see how our embeddings are treated:

# COMMAND ----------

unmasker.model.bert.encoder

# COMMAND ----------

# MAGIC %md
# MAGIC Let's just talk about these components:
# MAGIC - `BertAttention`: This component takes a sequence of vectors (let's call it `x`) as input and gets the `Q`, `K`, and `V` components via `query(x)`, `key(x)`, and `value(x)`, respectively. As these are all $768$-dimensional vectors - and are thereby multiplicatively compatible under transpose - the layer performs the attention computation with a few key modifications:
# MAGIC     - **Multi-Headed Attention:** This is talked about in lecture, but essentially $K$, $Q$, and $V$ are all slices up along the embedding dimension such that we get 12 trios with dimension $768/12 = 64$. This will give us 12 different attention results, and hence will allow the network to distribute attention in a variety of ways. At the end, just concatenate embedding-wise and you'll be back up to 768-features vectors.
# MAGIC     - **Masked Attention:** This is less useful for BERT but explains what the `attention_mask` input is doing. Essentially, it's an boolean "should-I-add-negative-infinity-to-the-attention" mask to keep the model from attending to things it shouldn't. For inference purposes, this is usually not important barring the presence of padding tokens. When using off-the-shelf pipelines for inference, you can ignore attention masks in most cases and can assume that the pipeline will take care of it.
# MAGIC     - **Residual Connections:** To help the network keep the token-level information propagating through the network (and to improve the overall gradient flow), most architectures add residual connections around the transformer components.
# MAGIC
# MAGIC - `BertSelfOutput -> BertIntermediate -> BertOutput`: These are all just token-level dense layers with non-linear activations and some `LayerNorm`/`Dropout` layers mixed in for regularization. Each element in the sequence is thereby ran through a MLP with dimension $768 \to 768 \to 3072 \to 768$ to a new representation.
# MAGIC
# MAGIC And... there are 12 of these, stacked one after the other! Not too bad, huh?

# COMMAND ----------

# MAGIC %md
# MAGIC <div><img src="imgs/bert-construction.png" width="800"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizing The Attention Mechanism In Action
# MAGIC
# MAGIC Recall that there are 12 `SelfAttention` layers and each one has 12 attention heads reasoning about a different properties of the sequence. As such, we can request the realized attention values computed at each `SelfAttention` layer:

# COMMAND ----------

import torch

string = "Hello Mr. Bert! How is it [MASK]?"
input_tensors = unmasker.preprocess(string)
embeddings = unmasker.model.bert.embeddings(input_tensors['input_ids'])
x = unmasker.model.bert.encoder(embeddings, input_tensors['attention_mask'], output_attentions=True)
## NOTE, you can also feed it in as an argument on model construction

print('', "From encoder.forward():", sep='\n')
for k,v in x.items():
    if type(v) in (tuple, list):
        print(f" > '{k}' : {torch.stack(v).shape}")
    else:
        print(f" > '{k}' : {v.shape}")


# COMMAND ----------

# MAGIC %md
# MAGIC As the transformer architecture largely avoids mixing semantics/position information outside of the attention mechanism. As such, you can claim that the attention localized at any head loosely considers the impact of the whole sequence on a particular sequence entry (aka token).
# MAGIC
# MAGIC To visualize this, we can use the [`BertViz` package](https://github.com/jessevig/bertviz) to display the attention associations from our last forward pass in an interactive grid! Please feel free to test this out with other input strings to see what changes.
# MAGIC - See what happens to the dimensionality when you increase the number of tokens.
# MAGIC - See what happens to the connections, and see if you see any patterns worth noting.
# MAGIC - Why do you think the CLS and SEP tokens get so much attention in many of the attention heads?

# COMMAND ----------

# IGNORE, WILL NOT WORK

from bertviz import model_view

import torch
from transformers import pipeline

string = "Hello Mr. Bert! [MASK] should be fun!"
input_ids = unmasker.tokenizer.encode(string)
input_tokens = unmasker.tokenizer.convert_ids_to_tokens(input_ids)

input_tensors = unmasker.preprocess(string)
embeddings = unmasker.model.bert.embeddings(input_tensors['input_ids'])
x = unmasker.model.bert.encoder(embeddings, input_tensors['attention_mask'], output_attentions=True)

model_view(x['attentions'], input_tokens)  # Display model view


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4. Wrapping Up
# MAGIC
# MAGIC At this point, we've shows the core intuitions of how these models reason about text:
# MAGIC
# MAGIC - Embed the semantics and positions of the tokens.
# MAGIC - Reason about the token components, mostly in isolation and with small and tight interfaces to consider the other tokens in the sequence.
# MAGIC
# MAGIC These several modifications are intuitive to understand and work well in practice, and just about every model we interact with will rely on this intuition. 
# MAGIC

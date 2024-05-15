---
title: "BPE tokenization"
date: 2024-05-12
categories: "deep_learning"
---

**Overview**

Byte-Pair Encoding (BPE) tokenization is a common technique to convert text to trainable input array before feeding it to a large language model (LLM). It is the tokenizer used by some of the most influential models such as GPT-4 and RoBERTa. Andrej Karpathy has [an excellent video][karpathy_tutorial] about this process. I will summarize and expand on his ideas in this writing. The idea of BPE is rather simple and is showcased by its Wikipedia page: suppose we have the following data to encode `aaabdaaabac`, we replace the most frequent adjacent pairs of bytes (`aa`) by a new vocabulary (`Z`), so it becomes (`ZabdZabac`). We can repeat this process again and again until we reach a certain vocabulary size, reducing the original data into chunks represented by the new vocabulary in the process. 

**Detailed Process**

First of all, BPE (Byte Pair Encoding) requires an array of byte representations of data. This can be achieved using the UTF-8 encoding scheme. UTF-8 is a character-level encoding process that transforms characters into 1 to 4 bytes, with the most common characters transformed into 1 byte and less common ones into up to 4 bytes. Just like `aaabdaaabac`, the transformed byte array is used as the basis of the TPE process where the most popular adjacent bytes are grouped into a new vocabulary. The Python codes below (by Karpathy) showcase this process.

{% highlight python %}
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokens = text.encode('utf-8')
tokens = list(map(int, tokens))
print("----")
print(text)
print("length: ", len(text)) # output: 533
print("----")
print(tokens) # output: [239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 18, ...]
print("length: ", len(tokens)) # output: 616
{% endhighlight %}

First, a text string is encoded into byte tokens using UTF-8, notice how the length of the encoded tokens is larger than the length of the original characters. This is because some characters such as `🅤` are encoded to more than 1 byte. In order to get which pairs are popular, we also need a function to establish the pairs' statistics within the data:

{% highlight python %}
def get_stats(ids):
  counts = {}
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

stats = get_stats(tokens)
print(sorted(((v, k) for k, v in stats.items()), reverse=True)) # output: [(20, (101, 32)), (15, (240, 159)), (12, (226, 128)), ...]
top_pair = max(stats, key=stats.get)
{% endhighlight %}

The above code registers each adjacent byte pair's frequency in the data, and list them in descending order. Here, the byte pair (101, 32) appears most frequently (20 times) in the text data. The pairs are the byte representation of `space` and `e`. 

Now, with the statistics of pairs, we can merge the most frequent pair to a new vocabulary:

{% highlight python %}
def merge(ids, pair, idx):
  """
  ids: data byte array (encoded by utf-8)
  pair: pairs of bytes to be merged into a new vocabulary
  idx: the idx of the new vocabulary for merged bytes
  """
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print("length: ", len(tokens2)) # output: 596
{% endhighlight %}

Here, we merged the most frequent pair (`(101, 32)`) into a new vocabulary (`256`). 256 is chosen because the bytes encoded using UTF-8 ranges from 0 to 255. Notice how the length diminished from `616` to `596`, as 20 instances have been merged. Such merges can of course be done repeatedly on a larger textual corpus: 

{% highlight python %}
text = # """<some longer text, full article of A programmer's introduction to Unicode (https://www.reedbeta.com/blog/programmers-intro-to-unicode/)>"""
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
len(tokens) # output: 24291

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy original list
merges = {}
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f'merging {pair} into a new token {idx}')
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print("tokens length: ", len(tokens))
print("ids length: ", len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}%")

# outputs:
# merging (101, 32) into a new token 256
# merging (105, 110) into a new token 257
# merging (115, 32) into a new token 258
# merging (116, 104) into a new token 259
# merging (101, 114) into a new token 260
# merging (116, 32) into a new token 261
# merging (99, 111) into a new token 262
# merging (226, 128) into a new token 263
# merging (44, 32) into a new token 264
# merging (97, 110) into a new token 265
# merging (111, 114) into a new token 266
# merging (100, 32) into a new token 267
# merging (97, 114) into a new token 268
# merging (101, 110) into a new token 269
# merging (257, 103) into a new token 270
# merging (262, 100) into a new token 271
# merging (121, 32) into a new token 272
# merging (259, 256) into a new token 273
# merging (97, 108) into a new token 274
# merging (111, 110) into a new token 275
# tokens length:  24291
# ids length:  19241
# compression ratio: 1.26%
{% endhighlight %}

Here, we have repeatedly merged adjacent byte pairs and increased our vocabulary size from 256 to 276, reducing the length of byte level data from 24291 to 19241 in the process. Such a reduction helps fit data into transformers' context window and achieves better empirical results compared to more granular byte arrays. In practice, the vocabulary size can reach up to ~50k (GPT-2 tokenizer) or even ~100k (GPT-4 tokenizer).

With the merges done, we are now able to encode the original byte data using the new vocabulary. The code below illustrates the encoding process:

{% highlight python %}
def encode(text):
  tokens = list(text.encode('utf-8'))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float('inf')))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

print(encode('hello world!')) # output: [104, 101, 108, 108, 111, 32, 119, 266, 108, 100, 33]
{% endhighlight %}

There are several things to notice here. First, `while len(tokens) >= 2` ensures that we have at least 2 tokens to merge, as otherwise we cannot merge. Second, `pair = min(stats, key=lambda p: merges.get(p, float('inf')))` takes the pair within our data (`stats`) ranked by its frequency in the merges dictionary (trained separately on a corpus of text). If the pair does not exist in the `merges`dictionary, we assign infinity to it so it wouldn't be picked up if any other pair exists in `merges`. If none of the pairs of our encoded data are in merges, all pairs will have infinity as value and `min` returns the first such pair, and we will exit the loop. Just like how `merges` is established, any hierarchy within the data pairs will preserve their structure. For example, if `ab` is labeled as `Z`, and `Zc` is labeled as `Z'`, `abc` will first be encoded in to `Zc` and then to `Z'`, just like in `merges`. 

In this example, `hello world!` has output `[104, 101, 108, 108, 111, 32, 119, 266, 108, 100, 33]` of length 11, while the UTF-8 encoding of `hello world!` is `[104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]` and has length 12. It is because our `merges` dictionary has combined `111 (o)` and `114 (r)` into `266 (or)`. In non-toy scenarios, the final encoding will be even shorter, using the gpt-2 tokenizer of `tiktoken` library, `hello world!` is encoded to `[31393 (hello), 995 ( world), 0 (!)]`. 


Now, we are ready to tackle the decoding process in the following code:

{% highlight python %}
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode('utf-8', errors='replace')
  return text
{% endhighlight %}

Here, the `ids` input is the encoded data (after merges), the line `tokens = b"".join(vocab[idx] for idx in ids)` gives us the concatenated byte representations of `ids`, where the new vocabulary within `merges` are simply concatenations of the original pair. For example, the byte representation of `or` is simply the byte representation of `o` concatenated with the byte representation of `r`. It should be noted that the code above assumes the iterator of `merges` follows order of insertion, as similar to encoding, there is a hierarchy within. Suppose we have `ord` somewhere in `merges` where the key pair is `(or, d)`, we need the byte representation of `or` to be estalished first in `vocab` before we can process `ord`. The insertion order ensures this is the case. Another detail of interest is the `errors='replace'` error handling. During the merging process, it is possible to create invalid UTF-8 byte sequences that do not follow the UTF-8 byte convention. `errors='replace'` will replace these bytes with the replacement token `�` for a more robust decoding. 

We can test our BPE with the following line and it should pass the assertion:

{% highlight python %}
assert decode(encode('hello world!')) == 'hello world!'
{% endhighlight %}

**GPT-2 Tokenizer**

In practice, GPT-2 and GPT-4 tokenizers are just the above technique applied on a larger scale. However, there are 2 important additions:

* Use of regular expression to first compartmentalize the input text before performing encoding. The following codes from the [gpt-2 encoder][gpt-2_encoder] showcase this:

  {% highlight python %}
  import regex as re
  <...>

  class Encoder: 
    def __init__(self, encoder, bpe_merges, errors='replace'):
      <...>
      self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    <...>

    def encode(self, text):
      <...>
      for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      <...>
  {% endhighlight %}

  Here, the regular expression pattern `r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""` is used to tokenize a given text string. Common contractions such as 's and 't are treated as individual tokens. Unicode letter strings with optional leading space (``` ?\p{L}+```), Unicode number strings with optional leading space (``` ?\p{N}+```), non-letter and non-number such as punctuations (``` ?[^\s\p{L}\p{N}]+```) are also tokenized, along with other patterns. Once compartmentalized, these strings can be UTF-8 encoded using `token.encode('utf-8')`.

  The reason why this is done is because without such compartmentalization, BPE tokenizer might "group many versions of common words like dog
  since they occur in many variations such as dog. dog!
  dog? . This results in a sub-optimal allocation of limited
  vocabulary slots and model capacity"[^1]. With such compartmentalization, strings like dog! will first be compartmentalized into dog and ! before BPE encoding, thus avoiding the above cases.


* Use of an additional layer `byte_encoder` on top of `UTF-8` before performing encoding. The following codes showcase this:

  {% highlight python %}
  <...>
  from functools import lru_cache
  
  @lru_cache()
  def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

  class Encoder: 
    def __init__(self, encoder, bpe_merges, errors='replace'):
      <...>
      self.byte_encoder = bytes_to_unicode()
      <...>

    <...>

    def encode(self, text):
      bpe_tokens = []
      for token in re.findall(self.pat, text):
        token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
      return bpe_tokens
  {% endhighlight %}

  This is a somewhat superfluous step which is only necessary when the BPE encoding happens on Unicode characters (as in the case of the GPT-2 encoder) instead of bytes. the `bytes_to_unicode()` function gives a static mapping of each byte from 0 - 255 to a unique Unicode character that is not a whitespace or control character, and then `self.bpe()` function does its merging job on the resulting characters instead of directly on bytes. It is an artificially confusing step that does not add to the algorithm.

Another thing to note for the GPT-2 tokenizer are the `encoder` and `bpe_merges` files during `__init__`. These files are similar to the `vocab` and `merges` files in our code above. In fact, when we do `tiktoken.get_encoding('gpt2')`, we are internally making requests to the following files: `https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json` and `https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe`. Taking a peek, we can see that `encoder.json` looks like the following:

```
{"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, ..., "\u0120informants": 50254, "\u0120gazed": 50255, "<|endoftext|>": 50256}
```
and `vocab.bpe` looks like the following:

```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he
e r
...
Com par
Ġampl ification
om inated
Ġreg ress
ĠColl ider
Ġinform ants
Ġg azed
```
Notice how there is a special token `<|endoftext|>: 50256` used to denote end of documents during training. Special tokens are not the result of BPE merges, but are rather artificial tokens that we can add to the tokenizer. As a result, they are also handled specially before BPE encoding. 

[karpathy_tutorial]: https://www.youtube.com/watch?v=zduSFxRajkE
[gpt-2_encoder]: https://github.com/openai/gpt-2/blob/master/src/encoder.py
[^1]: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
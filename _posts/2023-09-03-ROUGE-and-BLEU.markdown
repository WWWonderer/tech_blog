---
title: "ROUGE and BLEU"
date: 2023-09-03
categories: "NLP"
---
In natural language processing (NLP), evaluation metrics are often not as straightforward as in computer vision (CV), due to the different ways in a language to express the same meaning. Below are 2 popular metrics in NLP.

**ROUGE**

*ROUGE* (recall-oriented understudy for gisting evaluation) is mainly an algorithm for evaluating the quality of automatic machine summarizations. Its output is between 0 and 1, with values closer to 1 representing better summarizations. In the original paper, 4 types of ROUGE scores are presented: *ROUGE-N* (N-gram co-occurrence statistics), *ROUGE-L* (longest common subsequence), *ROUGE-W* (weighted longest common subsequence), and *ROUGE-S* (skip-bigram co-occurrence statistics). In this post only *ROUGE-N* and *ROUGE-L* will be discussed, as they are the most popular. *ROUGE-L* comes with 2 flavors: the sentence-level *ROUGE-L*, and the summary-level *ROUGE-Lsum*.

*ROUGE-N* is defined as follow in the [original paper][rouge_paper]:


$$
\begin{aligned}
\text{ROUGE-N}(candidate, reference) &= \frac{\sum_{r_i \in reference} \sum_{\text{N-gram} \in r_i} Count(\text{N-gram}, candidate)}{\sum_{r_i \in reference} Count(\text{N-gram}, r_i)}
\end{aligned}
$$
where $r_i$ are sentences in the reference document. Since the denominator is the reference, this is a recall based measure. In [HuggingFace implementation][huggingface_rouge], precision (using candidate as denominator) and f1 scores ($2(\frac{precision \cdot recall}{precision + recall})$)are also computed. 

*ROUGE-L* signifies the longest common subsequence and is defined as follow:

$$\begin{align}
\text{ROUGE-L}(c, r) &= \frac{(1+\beta^2)R_{lcs}(c, r)P_{lcs}(c, r)}{R_{lcs}(c, r) + \beta^2P_{lcs}(c, r)} \\
\\
R_{lcs} &= \begin{cases}
            \frac{LCS(r, c)}{|r|} & \text{if sentence-level (ROUGE-L)} \\
            \frac{\sum_{r_i=i}^{\text{num-sentences}}LCS_{\cup}(r_i, c)}{|r|} & \text{if summary level (ROUGE-Lsum)}
           \end{cases}\\
\\
P_{lcs} &= \begin{cases}
            \frac{LCS(r, c)}{|c|} & \text{if sentence-level (ROUGE-L)} \\
            \frac{\sum_{r_i=i}^{\text{num sentences}}LCS_{\cup}(r_i, c)}{|c|} & \text{if summary-level (ROUGE-Lsum)}
           \end{cases}
\end{align}$$


here, $c$ signifies the candidate summary and $r$ the reference summary, $\|c\|$ signifies the number of words in $c$, $\|r\|$ the number of words in $r$. Since ROUGE is more associated with recall, the $\beta$ in the F-score is usually set to a higher value such as 2. The longest common subsequence ($LCS(r,c)$) is defined as follow:

A sequence $Z = [z_1, z_2, ..., z_n]$ is a subsequence of
another sequence $X = [x_1, x_2, ..., x_m]$, if there exists a
strict increasing sequence $[i_1, i_2, ..., i_k]$ of indices of
$X$ such that for all $j = 1, 2, ..., k$, we have $x_{i_j} = z_j$. Given two sequences $r$ and $c$, the longest common subsequence ($LCS(r, c)$) of $r$ and $c$ is a common subsequence with maximum length. $LCS_\cup$ signifies the union of different $LCS$. For example, if $r_i = w_1 w_2 w_3 w_4 w_5$, and $c$ contains two sentences: $c_1 = w_1 w_2 w_6 w_7 w_8$ and $c2 = w_1 w_3 w_8 w_9 w_5$, then the longest common subsequence of $r_i$ and $c_1$ is $w_1 w_2$ and the longest common subsequence of $r_i$ and $c_2$ is $w_1 w_3 w_5$. The union longest common subsequence of $r_i$, $c_1$, and $c_2$ is $w_1 w_2 w_3 w_5$ and $LCS_\cup(r_i, c) = 4$ ([Lin, 2004][rouge_paper]).

The [google-research/huggingface][rouge_source] implementation of ROUGE is as follow:

{% highlight python %}
def score(self, target, prediction):
  """Calculates rouge scores between the target and prediction.

  Args:
    target: Text containing the target (ground truth) text,
    or if a list
    prediction: Text containing the predicted text.
  Returns:
    A dict mapping each rouge type to a Score object.
  Raises:
    ValueError: If an invalid rouge type is encountered.
  """

  # Pre-compute target tokens and prediction tokens for use by different
  # types, except if only "rougeLsum" is requested.
  if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
    target_tokens = None
    prediction_tokens = None
  else:
    target_tokens = self._tokenizer.tokenize(target)
    prediction_tokens = self._tokenizer.tokenize(prediction)
  result = {}

  for rouge_type in self.rouge_types:
    if rouge_type == "rougeL":
      # Rouge from longest common subsequences.
      scores = _score_lcs(target_tokens, prediction_tokens)
    elif rouge_type == "rougeLsum":
      # Note: Does not support multi-line text.
      def get_sents(text):
        if self._split_summaries:
          sents = nltk.sent_tokenize(text)
        else:
          # Assume sentences are separated by newline.
          sents = six.ensure_str(text).split("\n")
        sents = [x for x in sents if len(x)]
        return sents

      target_tokens_list = [
          self._tokenizer.tokenize(s) for s in get_sents(target)]
      prediction_tokens_list = [
          self._tokenizer.tokenize(s) for s in get_sents(prediction)]

      scores = _summary_level_lcs(target_tokens_list,
                                  prediction_tokens_list)
    elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
      # Rouge from n-grams.
      n = int(rouge_type[5:])
      if n <= 0:
        raise ValueError("rougen requires positive n: %s" % rouge_type)
      target_ngrams = _create_ngrams(target_tokens, n)
      prediction_ngrams = _create_ngrams(prediction_tokens, n)
      scores = _score_ngrams(target_ngrams, prediction_ngrams)
    else:
      raise ValueError("Invalid rouge type: %s" % rouge_type)
    result[rouge_type] = scores

  return result
{% endhighlight %}


**BLEU**

BLEU (bilingual evaluation understudy) is mainly an algorithm for evaluating the quality of machine translations. Its output is between 0 and 1, with values closer to 1 representing more similar texts. In practice, a score of 1 is unattainable as it indicates a perfect match between machine and human translated texts. A score of 0.6 to 0.7 is often the best one can get.

BLEU score can be seen as roughly the weighted geometric average precision of 1, 2, ..., N N-gram precision scores, which is then slightly adjusted with a brevity prior. In mathematical notation, we have:

$$\begin{aligned}
\text{BLEU} (N) &= \text{N-gram precision}(N) \cdot \text{brevity penalty}\\
\\
\text{N-gram precision}(N) &= (p_1)^{w_1} \cdot (p_2)^{w_2}  ...  (p_N)^{w_N} \\
        &= \prod_{n=1}^N p_n^{w_n} \\
        &= e^{\sum_{n=1}^N w_n \text{log}(p_n)} \\
\\
p_n &= \frac{\text{total number of N-gram in both translation and reference}}{\text{total number of N-gram in translation}} \\
\\
\text{brevity penalty} &= \begin{cases} 
                        1 & \text{if c $\ge$ r}  \\ 
                        e^{(1 - \frac{r}{c})} & \text{if c $\lt$ r}
                        \end{cases}
\end{aligned}$$

Here, $p_1$, $p_2$, ..., $p_N$ signifies the 1-gram, 2-gram, ... N-gram precision scores, with respective weights of $w_1$, $w_2$, ..., $w_N$ in the geometric mean. $c$ is the length of the candidate corpus and $r$ is the effective reference corpus length. As the calculation of precision favors shorter sentences, brevity penalty is introduced to balanced out this effect, as shorter candidates get penalized with a prior less than 1. In practice, N = 4 with all weights being $\frac{1}{4}$ is often used. 

Below is the [tensorflow/huggingface implementation][bleu_source] of BLEU score:

{% highlight python %}
def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order) # | means set union
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts # & means set intersection
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)
{% endhighlight %}





[bleu_source]: https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py
[huggingface_rouge]: https://huggingface.co/spaces/evaluate-metric/rouge
[rouge_paper]: https://aclanthology.org/W04-1013.pdf
[rouge_source]: https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py

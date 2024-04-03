---
title: "SA-IS algorithm"
date: 2024-04-02
categories: "algorithms"
---

**What is SA-IS?**

SA-IS stands for Suffix Array Induced Sorting. It is an algorithm to construct suffix arrays in linear time. Suffix arrays are all the suffixes of a sequence ranked w.r.t. a certain order. For example, in the string "banana", all the suffixes are "banana", "anana", "nana", "ana", "na", and "a", and the corresponding suffix array is [5, 3, 1, 0, 4, 2] where each number corresponds to the index of the corresponding suffix ranked in alphabetical order (i.e. 5 is the index of suffix "a", ranked 1st, "3" is the index of suffix "ana", ranked 2nd). Suffix arrays are important because they allow for quick binary search (O(logN)) of patterns within the original array and form the foundation of certain more sophisticated operations. They are especially important when the data is big such as in the case of big textual corpus in NLP or large DNA sequencing in bioinformatics. 

**Understanding SA-IS**

I find SA-IS to be one of the most difficult algorithms that I have encountered, both in terms of its conceptual ingenuity and the implementation that I came across. The original paper can be found [here][original_paper], but a more accessible paper explaining it can be found [here][second_paper], the implementation is found [here][implementation]. 

The essence of SA-IS are the 2 following ideas:
1. If a sequence is fully ascending (i.e. aaabc) or fully descending (i.e. cbaaa), then construction of the suffix array will be simple. What make it hard are the "knots" within the sequence where the ascending/descending trend switches (i.e. cbaaabc, where 'a' changes the trend from descending to ascending). If we can rank the "knots", then we can rank the rest of the sequence easily. 
2. In order to rank the "knots", we can divide the original sequence using substrings containing these "knots" and label them using a new alphabet, a process called lexical naming. This way, we reduce the original problem into a smaller problem recursively, where the sum of these recursions are bounded by the original sequence's length.

There are several important definitions before we can formalize these ideas, to follow along, one should first read the "A Close look of SA_IS" section of the [2nd paper][second_paper], especially figures 3, 4, 5, 6, 7. One should understand what the 'w' substring means and what are the 3 steps of the algorithm. The paper also have a [supplementary section][second_paper_appendix] where proofs for why the algorithm works can be found. I won't repeat them here as it would be repetitive and a waste of my time. What I will present however is what I found to be fuzzy even after reading the papers, mainly the recursion part, and how to rank the 'w' substring part. I will do so using careful debugging of the implementation. 

**Implementation**

The [implementation][implementation] I came across is written in Rust, a wonderful language that I didn't have much experience on. I used VSCode and the Rust-Analyser extension to debug Rust on Windows. The MS C/C++ extension also has to be installed for Rust-Analyser's debugger to work as it provides the necessary debugging interfaces for LLVM-generated binaries shared by both languages. Once this is done, we can setup a launch configuration using `rust-analyzer: Generate launch configuration` of VSCode command palette. Specific to this implementation, we also need to change the optimization level from `opt-level=3` to `opt-level=0` in Cargo.toml. Otherwise, the debugger will skip quite a lot of checkpoints. The main implementation of the SA-IS algorithm is in the `fn sais<T>(sa: &mut [u64], stypes: &mut SuffixTypes, bins: &mut Bins, text: &T)` function of `table.rs`. I created an example string "lartistartist" and will use it to highlight all the phases of the suffix array creation. The array creation can be divided into 2 big steps: 1. sorting the 'w' substrings 2. sorting the rest of the substrings. Each big step can be further divided into smaller substeps. Below are the codes and their effects on the example string.


> | String Name  | Trend   |
> |--------------|-----------------|
> | lartistartist| DVADVADVADVAD   |
>
> where D means descending, V means valley, A means ascending. Refer to paper 2 above.

*SORTING THE WSTRINGS*
* Step 1: create suffix array (sa) partitioned with bins and pointers. This forms the foundation of bucket sort that we will use later. In "lartistartist", the buckets will be 'a', 'i', 'l', 'r', 's' and 't'.
    {% highlight rust %}
for v in sa.iter_mut() {
    *v = 0;
}
stypes.compute(text);
bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
bins.find_tail_pointers();
    {% endhighlight %}

    
    a  i  l r  s  t
    __|__|_|__|__|____|
    0  2  4 5  7  9

* Step 2: insert valley suffixes following the bucket sort process described in step 0 of paper 2. This is only a guess, based on the assumption that longer substrings are lower in ranking.

    {% highlight rust %}
// Insert the valley suffixes.
for (i, c) in text.char_indices().map(|v| v.idx_char()) {
    if stypes.is_valley(i as u64) {
        bins.tail_insert(sa, i as u64, c);
    }
}
    {% endhighlight %}
    
    a   i    l r  s  t
    7 1|10 4|_|__|__|____|
    0   2    4 5  7  9

* Step 3: insert descending suffixes based on the 'w' substrings and the induced descending suffixes themselves, following the bucket sort process described in step 1 of paper 2.
    {% highlight rust %}
// Now find the start of each bin.
bins.find_head_pointers();

// Insert the descending suffixes.
let (lasti, lastc) = text.prev(text.len());
if stypes.is_desc(lasti) {
    bins.head_insert(sa, lasti, lastc);
}
for i in 0..sa.len() {
    let sufi = sa[i];
    if sufi > 0 {
        let (lasti, lastc) = text.prev(sufi);
        if stypes.is_desc(lasti) {
            bins.head_insert(sa, lasti, lastc);
        }
    }
}
    {% endhighlight %}

    a   i    l r  s  t
    7 1|10 4|0|__|__|12 6 9 3|
    0   2    4 5  7  9

* Step 4: insert ascending suffixes based on the descending suffixes and the induced ascending suffixes themselves, following the bucket sort process described in step 2 of paper 2. These first 4 substeps are almost identical to the substeps of the 2nd major step (sorting the rest of the substrings). The only difference being these are conditioned on a guessed position of 'w' substrings, while in the 2nd major step the position of the 'w' substrings are known to be correct.
    {% highlight rust %}
// ... and the find the end of each bin.
bins.find_tail_pointers();

// Insert the ascending suffixes.
for i in (0..sa.len()).rev() {
    let sufi = sa[i];
    if sufi > 0 {
        let (lasti, lastc) = text.prev(sufi);
        if stypes.is_asc(lasti) {
            bins.tail_insert(sa, lasti, lastc);
        }
    }
}
    {% endhighlight %}
    a   i    l r   s    t
    7 1|10 4|0|8 2|11 5|12 6 9 3|
    0   2    4 5   7    9

* Step 5: find and move all wstrings to the beginning (in this example it is unchanged as all(num_wstrs = 4) the wstrings are already at the beginning after step 1). This is the start of a series of manipulations (steps 5 - 10) involving lexical naming and potential recursions, with the goal of making sure the 'w' substrings are correctly ranked in the end.
    {% highlight rust %}
// Find and move all wstrings to the beginning of `sa`.
let mut num_wstrs = 0u64;
for i in 0..sa.len() {
    let sufi = sa[i];
    if stypes.is_valley(sufi) {
        sa[num_wstrs as usize] = sufi;
        num_wstrs += 1;
    }
}
// This check is necessary because we don't have a sentinel, which would
// normally guarantee at least one wstring.
if num_wstrs == 0 {
    num_wstrs = 1;
}
    {% endhighlight %}
    a   i    l r   s    t
    7 1|10 4|0|8 2|11 5|12 6 9 3|
    0   2    4 5   7    9

* Step 6: replace all non-wstrings with max value(m), then put the associated lexical names of the wstring at index 'sa[num_wstrs + cur_sufi / 2]', where 'cur_sufi' is the index of the wstring in the original text. This step basically puts lexical names for the wstrings from left to right.

    {% highlight rust %}
let mut prev_sufi = 0u64; // the first suffix can never be a valley
let mut name = 0u64;
// We set our "name buffer" to be max u64 values. Since there are at
// most n/2 wstrings, a name can never be greater than n/2.
for i in num_wstrs..(sa.len() as u64) {
    sa[i as usize] = u64::MAX;
}
for i in 0..num_wstrs {
    let cur_sufi = sa[i as usize];
    if prev_sufi == 0 || !text.wstring_equal(stypes, cur_sufi, prev_sufi) {
        println!("prev_sufi: {}, cur_sufi: {}", prev_sufi, cur_sufi);
        name += 1;
        prev_sufi = cur_sufi;
    }
    // This divide-by-2 trick only works because it's impossible to have
    // two wstrings start at adjacent locations (they must at least be
    // separated by a single descending character).
    sa[(num_wstrs + (cur_sufi / 2)) as usize] = name - 1;
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|m 2|0 m|1 m m m|
    0   2    4 5   7   9

* Step 7: smush the inserted lexical names at the end of the array. This is a preparatory step for the potential split and recursion of the next step. 
    {% highlight rust %}
// We've inserted the lexical names into the latter half of the suffix
// array, but it's sparse. so let's smush them all up to the end.
let mut j = sa.len() as u64 - 1;
for i in (num_wstrs..(sa.len() as u64)).rev() {
    if sa[i as usize] != u64::MAX {
        sa[j as usize] = sa[i as usize];
        j -= 1;
    }
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|m 2|0 m|0 2 0 1|
    0   2    4 5   7   9

* Step 8: (OPTIONAL) if there are repeated lexical name (indicating identical wstrings), split and recurse. In our example, we do have repeated lexical names ('0'), so we enter the recursion code inside the "if" block, the "else" block will be explained later.
{% highlight rust %}
// If we have fewer names than wstrings, then there are at least 2
// equivalent wstrings, which means we need to recurse and sort them.
if name < num_wstrs {
    let split_at = sa.len() - (num_wstrs as usize);
    let (r_sa, r_text) = sa.split_at_mut(split_at);
    sais(&mut r_sa[..num_wstrs as usize], stypes, bins, &LexNames(r_text));
    stypes.compute(text);
} else {
    for i in 0..num_wstrs {
        let reducedi = sa[((sa.len() as u64) - num_wstrs + i) as usize];
        sa[reducedi as usize] = i;
    }
}
{% endhighlight %}
    In this example, we have: 
    |wstring index|wstring       |suffix        |lexical name|
    |-------------|--------------|--------------|------------|
    |7            |arti          |artist        |0           |
    |1            |arti          |artistartist  |0           |
    |10           |ist<sentinel> |ist           |1           |
    |4            |ista          |istartist     |2           |

    The recursion enters by spliting the original sa and take its head (containing wstrings 7 1 10 4) as the new sa and its tail (containing lexical names 0 2 0 1) as the new text, discarding the middle (0 m 2 0 m). The lexical names keep the order of the original text, this guarantees that sorting these names is equivalent to sorting the original suffixes.
    a   i    l r   s           t
    7 1|10 4|0|m 2|0 m    |    0 2 0 1| //split at 'sa.len() - num_wstrs'
    0   2    4 5   7           9

    The recursion returns (2 0 3 1) which is the suffix array of the lexical names (0 2 0 1). This updates the original sa to:
    a   i   l r   s   t
    2 0|3 1|0|m 2|0 m|0 2 0 1|
    0   2   4 5   7   9

* Step 9: replace the lexical names with their corresponding suffix index in the original text
    {% highlight rust %}
// Re-calibrate the bins by finding their sizes and the end of each bin.
bins.find_sizes(text.char_indices().map(|c| c.idx_char().1));
bins.find_tail_pointers();

// Replace the lexical names with their corresponding suffix index in the
// original text.
let mut j = sa.len() - (num_wstrs as usize);
for (i, _) in text.char_indices().map(|v| v.idx_char()) {
    if stypes.is_valley(i as u64) {
        sa[j] = i as u64;
        j += 1;
    }
}
    {% endhighlight %}
    a   i   l r   s   t
    2 0|3 1|0|m 2|0 m|1 4 7 10|
    0   2   4 5   7   9

* Step 10: map the suffix indices from the reduced text to suffix indices in the original text using the information we got in step 9, and zero out everything after the wstrings. At this point, we can be sure that the wstrings' positions are correct.
    {% highlight rust %}
// And now map the suffix indices from the reduced text to suffix
// indices in the original text. Remember, `sa[i]` yields a lexical name.
// So all we have to do is get the suffix index of the original text for
// that lexical name (which was made possible in the loop above).
//
// In other words, this sets the suffix indices of only the wstrings.
for i in 0..num_wstrs {
    let sufi = sa[i as usize];
    sa[i as usize] = sa[(sa.len() as u64 - num_wstrs + sufi) as usize];
}
// Now zero out everything after the wstrs.
for i in num_wstrs..(sa.len() as u64) {
    sa[i as usize] = 0;
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|0 0|0 0|0 0 0 0|
    0   2    4 5   7   9

*SORTING THE REST OF THE SUBSTRINGS*
* Step 11: insert the valley suffixes again using bucket sort, this is different from step 2 as this time we know that the wstrings are sorted and not guessed. In this example, the suffix array doesn't change as all the wstrings are at the start of the alphabetical order.
    {% highlight rust %}
// Insert the valley suffixes and zero out everything else..
for i in (0..num_wstrs).rev() {
    let sufi = sa[i as usize];
    sa[i as usize] = 0;
    bins.tail_insert(sa, sufi, text.char_at(sufi));
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|0 0|0 0|0 0 0 0|
    0   2    4 5   7   9

* Step 12: insert the descending suffixes again.
    {% highlight rust %}
// Now find the start of each bin.
bins.find_head_pointers();

// Insert the descending suffixes.
let (lasti, lastc) = text.prev(text.len());
if stypes.is_desc(lasti) {
    bins.head_insert(sa, lasti, lastc);
}
for i in 0..sa.len() {
    let sufi = sa[i];
    if sufi > 0 {
        let (lasti, lastc) = text.prev(sufi);
        if stypes.is_desc(lasti) {
            bins.head_insert(sa, lasti, lastc);
        }
    }
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|0 0|0 0|12 6 9 3|
    0   2    4 5   7   9

* Step 13: insert the ascending suffixes again. This gives the final suffix array [7, 1, 10, 4, 0, 8, 2, 11, 5, 12, 6, 9, 3] for the string "lartistartist" in linear time.
    {% highlight rust %}
// ... and find the end of each bin again.
bins.find_tail_pointers();

// Insert the ascending suffixes.
for i in (0..sa.len()).rev() {
    let sufi = sa[i];
    if sufi > 0 {
        let (lasti, lastc) = text.prev(sufi);
        if stypes.is_asc(lasti) {
            bins.tail_insert(sa, lasti, lastc);
        }
    }
}
    {% endhighlight %}
    a   i    l r   s   t
    7 1|10 4|0|8 2|11 5|12 6 9 3|
    0   2    4 5   7   9

And finally, the SA-IS algorithm completes its mission.

What is left to discuss is the base case of the recursion: when there are no duplicate lexical names or identical wstrings in step 8. In this case, we enter the else block of step 8:
{% highlight rust %}
for i in 0..num_wstrs {
    let reducedi = sa[((sa.len() as u64) - num_wstrs + i) as usize];
    sa[reducedi as usize] = i;
}
{% endhighlight %}
This code basically returns us the lexical names of the non-duplicate wstrings from left to right, mushed at the right of the suffix array. This will ensure that the lexical names' suffix array is in correct order by the property of bucket sort and wstring itself, as if the wstrings starts with 2 different characters, bucket sort ensures the order, and if they start with the same character but are not identical, the property of wstring ensures the longer wstring's character corresponding to the last character of the shorter wstring is descending, therefore ranking after the shorter wstring. 


[original_paper]: https://local.ugene.unipro.ru/tracker/secure/attachment/12144/Linear%20Suffix%20Array%20Construction%20by%20Almost%20Pure%20Induced-Sorting.pdf
[second_paper]: http://bib.oxfordjournals.org/content/15/2/138.full.pdf
[implementation]: https://github.com/google-research/deduplicate-text-datasets
[second_paper_appendix]: https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bib/15/2/10.1093/bib/bbt081/2/bbt081-BiB2013_sup.pdf?Expires=1715103374&Signature=zc9m02cII~NZtxlkdocscn4Gd1iKYpfXST-g1Nv-45A05M8rGIm6wQB6L21gwu0~KG-9w29nA9qMEeebUDd93IV5PN2hyTnd0yDTZQbJYvhvsQbjYw499sruyXbVbGQBoN0xajKARfY6-P6OPcqEVOBKU4-vBGI1l0BOIvXswCgZuzh0TeKX0hCvNbWHKpg09sN0zQIdP~ekPqHm1ovYLMcjQoMrJLwdIgyZYSDQs-HBTKlGCaeMNTZTYrCrdFcprZIt2U8ncA8KrR8xvQKrz2gjDZe0XG6t0jCZrAgcjITC0zixR34z99rMhG06jmclJltONvwXOCvLMsiGT7lpLQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA
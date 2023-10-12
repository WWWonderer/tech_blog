---
title: "Trafilatura"
date: 2023-10-10
categories: "information_retrieval"
---

**What is trafilatura**

I was trying to web scrape and to process google search results, but encountered the following problem: *how to effectively capture the main content of a website and separate it from other things present on the website such as ads, links to other content, etc.* For a human this is straightforward as the main content typically resides in the middle of the website and is visually dominant, but for an automated process that treats raw html as input, this is not easy at all. After some research I found [trafilatura][trafilatura_github], a Python package to do just that. This package can be easily pip installed, has a simple to use API, and appears to be actively maintained at the time of this post.

**API**

The API of trafilatura is quite simple. Say we want to extract the main content of the following website: [https://github.blog/2019-03-29-leader-spotlight-erin-spiceland](https://github.blog/2019-03-29-leader-spotlight-erin-spiceland), we can simply do:

{% highlight python %}
import trafilatura

downloaded = trafilatura.fetch_url('https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/')
main_text = trafilatura.extract(downloaded) # main_text will be the main content of the website in text format
{% endhighlight %}

We can also use trafilatura to return other formats: 

{% highlight python %}
...
main_text_xml = trafilatura.extract(downloaded, output_format='xml') # main_text_xml will contain the main content in xml format.
{% endhighlight %}

Apart from `extract`, several other methods are useful as well:

* *bare_extraction* returns a dictionary including both information about the main content and the metadata. An example of returned dictionary is:

    ```
    {'title': 'Leader spotlight: Erin Spiceland',
    'author': 'Jessica Rudder',
    'url': 'https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/',
    'hostname': 'github.blog',
    'description': 'We’re spending Women’s History Month with women leaders who are making history every day in the tech community.',
    'sitename': 'The GitHub Blog',
    'date': '2019-03-29',
    'categories': [],
    'tags': [],
    'fingerprint': None,
    'id': None,
    'license': None,
    'body': None,
    'comments': '',
    'commentsbody': None,
    'raw_text': None,
    'text': '... (main content)',
    'language': None,
    'image': 'https://github.blog/wp-content/uploads/2019/03/Erin_FB.png?fit=4801%2C2521',
    'pagetype': 'article'}
    ```

* *extract_metadata* returns a metadata object which can be converted to a dictionary. An example is:

    ```
    {'title': 'Leader spotlight: Erin Spiceland',
    'author': 'Jessica Rudder',
    'url': 'https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/',
    'hostname': 'github.blog',
    'description': 'We’re spending Women’s History Month with women leaders who are making history every day in the tech community.',
    'sitename': 'The GitHub Blog',
    'date': '2019-03-29',
    'categories': [],
    'tags': [],
    'fingerprint': None,
    'id': None,
    'license': None,
    'body': None,
    'comments': None,
    'commentsbody': None,
    'raw_text': None,
    'text': None,
    'language': None,
    'image': 'https://github.blog/wp-content/uploads/2019/03/Erin_FB.png?fit=4801%2C2521',
    'pagetype': 'article'}
    ```

* *baseline* returns a 3 tuple of `lxml_object, text, length` where `lxml_object` is nothing but `text` wrapped inside a pair of <body><p> and </p></body> tags, and length is the length of the `text`. It is used when *extract* and trafilatura's wrapper of fallback libraries *readability-lxml* and *jusText* fails, and is purposed to return all textual tags aggregated in a single string.


**How does it work**
        
The inner workings of trafilatura is published by its author in a [2021 paper][trafilatura_paper]. In a nutshell, the software acts on raw html through XPath expressions and perform actions in the order of the following 2 perspectives:
1. **negative perpective:** *"it excludes unwanted parts of the HTML code (e.g. <div class="nav">)"*
2. **positive perspective (centering on desirable content):** *"the selected nodes of the HTML tree are then processed, i.e. checked for relevance (notably by element type, text length and link density) and simplified as to their HTML structure."*

The core code of software resides in its *core.py* file where the [*extract_content*][extract_content] function is the key for both perspectives. It first removes unwanted parts from HTML tree using the [*prune_unwanted_sections*][prune_unwanted_sections] function using XPaths created in the *xpaths.py* file, then processes the remaining parts with the [*handle_textelem*][handle_textelem] function, where different elements such as *list* or *p* are processed. 

For the removal part, a sample xpath is:
{% highlight xpath %}
OVERALL_DISCARD_XPATH = [
    # navigation + footers, news outlets related posts, sharing, jp-post-flair jp-relatedposts
    '''.//*[(self::div or self::item or self::list
             or self::p or self::section or self::span)][
    contains(translate(@id, "F","f"), "footer") or contains(translate(@class, "F","f"), "footer")
    or contains(@id, "related") or contains(translate(@class, "R", "r"), "related") or
    contains(@id, "viral") or contains(@class, "viral") or
    starts-with(@id, "shar") or starts-with(@class, "shar") or
    contains(@class, "share-") or
    contains(translate(@id, "S", "s"), "share") or
    contains(@id, "social") or contains(@class, "social") or contains(@class, "sociable") or
    contains(@id, "syndication") or contains(@class, "syndication") or
    starts-with(@id, "jp-") or starts-with(@id, "dpsp-content") or
    contains(@class, "embedded") or contains(@class, "embed")
    or contains(@id, "newsletter") or contains(@class, "newsletter")
    or contains(@class, "subnav") or
    contains(@id, "cookie") or contains(@class, "cookie") or contains(@id, "tags")
    or contains(@class, "tags")  or contains(@id, "sidebar") or
    contains(@class, "sidebar") or contains(@id, "banner") or contains(@class, "banner")
    or contains(@class, "meta") or
    contains(@id, "menu") or contains(@class, "menu") or
    contains(translate(@id, "N", "n"), "nav") or contains(translate(@role, "N", "n"), "nav")
    or starts-with(@class, "nav") or contains(translate(@class, "N", "n"), "navigation") or
    contains(@class, "navbar") or contains(@class, "navbox") or starts-with(@class, "post-nav")
    or contains(@id, "breadcrumb") or contains(@class, "breadcrumb") or
    contains(@id, "bread-crumb") or contains(@class, "bread-crumb") or
    contains(@id, "author") or contains(@class, "author") or
    contains(@id, "button") or contains(@class, "button")
    or contains(translate(@class, "B", "b"), "byline")
    or contains(@class, "rating") or starts-with(@class, "widget") or
    contains(@class, "attachment") or contains(@class, "timestamp") or
    contains(@class, "user-info") or contains(@class, "user-profile") or
    contains(@class, "-ad-") or contains(@class, "-icon")
    or contains(@class, "article-infos") or
    contains(translate(@class, "I", "i"), "infoline")
    or contains(@data-component, "MostPopularStories")
    or contains(@class, "options")
    or contains(@class, "consent") or contains(@class, "modal-content")
    or contains(@class, "paid-content") or contains(@class, "paidcontent")
    or contains(@id, "premium-") or contains(@id, "paywall")
    or contains(@class, "obfuscated") or contains(@class, "blurred")
    or contains(@class, " ad ")
    or contains(@class, "next-post")
    or contains(@class, "message-container") or contains(@id, "message_container")
    or contains(@class, "yin") or contains(@class, "zlylin") or
    contains(@class, "xg1") or contains(@id, "bmdh")
    or @data-lp-replacement-content]''',

    # comment debris + hidden parts
    '''.//*[@class="comments-title" or contains(@class, "comments-title") or
    contains(@class, "nocomments") or starts-with(@id, "reply-") or starts-with(@class, "reply-") or
    contains(@class, "-reply-") or contains(@class, "message")
    or contains(@id, "akismet") or contains(@class, "akismet") or
    starts-with(@class, "hide-") or contains(@class, "hide-print") or contains(@id, "hidden")
    or contains(@style, "hidden") or contains(@hidden, "hidden") or contains(@class, "noprint")
    or contains(@style, "display:none") or contains(@class, " hidden") or @aria-hidden="true"
    or contains(@class, "notloaded")]''',
]
{% endhighlight %}
which can then be used to remove html node as shown in the following pseudocode:
{% highlight python %}
for expr in OVERALL_DISCARD_XPATH:
    for subtree in tree.xpath(expr) # where tree is the source HTML
        # do some processing
        # ...
        subtree.getparent().remove(subtree)
{% endhighlight %}

For processing the remaining parts, the [*handle_textelem*][handle_textelem] function has different sub-functions for different tags. As an example, the [*handle_paragraphs*][handle_paragraphs] function takes in the paragraph and a list of tags, filters out the sub-elements of the paragraph according to these tags, and then process them:

{% highlight python %}
for child in element.iter('*'): # where element is the paragraph
    if child.tag not in potential_tags and child.tag != 'done':
        LOGGER.debug('unexpected in p: %s %s %s', child.tag, child.text, child.tail)
        continue
    # process child
    # ...
{% endhighlight %}

The list of tags is defined in *settings.py* and contains the following:

{% highlight python %}
TAG_CATALOG = frozenset(['blockquote', 'code', 'del', 'head', 'hi', 'lb', 'list', 'p', 'pre', 'quote'])
{% endhighlight %}







[trafilatura_github]: https://github.com/adbar/trafilatura
[trafilatura_paper]: https://aclanthology.org/2021.acl-demo.15.pdf
[extract_content]: https://github.com/adbar/trafilatura/blob/ca66992640a9dc8aaf5f43897f4554ec6dbcf2eb/trafilatura/core.py#L504
[prune_unwanted_sections]: https://github.com/adbar/trafilatura/blob/ca66992640a9dc8aaf5f43897f4554ec6dbcf2eb/trafilatura/core.py#L477
[handle_textelem]: https://github.com/adbar/trafilatura/blob/ca66992640a9dc8aaf5f43897f4554ec6dbcf2eb/trafilatura/core.py#L426
[handle_paragraphs]: https://github.com/adbar/trafilatura/blob/ca66992640a9dc8aaf5f43897f4554ec6dbcf2eb/trafilatura/core.py#L252
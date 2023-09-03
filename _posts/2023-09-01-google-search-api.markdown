---
title: "Google search API"
date: 2023-09-01
categories: "information_retrieval"
---

**What is Google search API**

[Google Custom Search JSON API][google_search_api_url] allows you to programmatically search the web using Google. It will return similar (but slightly different) results as typical Google search from the browser in JSON format.

**Pricing**

Google search API is free up until 100 search queries per day. Additional queries cost $5 per 1000 queries, up to 10k queries per day. A [billing][google_cloud_billing] account must be set up if free quota is exceeded.


**API key and programmable search engine**

We need 2 things to use Google search API:
* an **API key**, which is a way to identify your client to Google. 

* a [programmable search engine][programmable_search_engine] and its associated  **CX key**, which is a way to identify your custom search engine to the API. Multiple custom search engines can be created, each with a different configuration. 

**Example usage**

To use the API, one should send a HTTP Get request to the following address with one's parameters:

```https://www.googleapis.com/customsearch/v1?[parameters]```

3 parameters are necessary: **API key**, **CX key**, **query**. A typical example in Python is the following:

{%highlight python %}import requests
api_url = "https://www.googleapis.com/customsearch/v1?key={API_key}&cx={CX_key}&q={search_query}&num={number_of_results}"
results = requests.get(api_url)
{% endhighlight %}

**Returned JSON**

For the query "Why always me?" and "5" results, the returned json for me was:
{%highlight json %}
{
  "kind": "customsearch#search",
  "url": {
    "type": "application/json",
    "template": "https://www.googleapis.com/customsearch/v1?q={searchTerms}&num={count?}&start={startIndex?}&lr={language?}&safe={safe?}&cx={cx?}&sort={sort?}&filter={filter?}&gl={gl?}&cr={cr?}&googlehost={googleHost?}&c2coff={disableCnTwTranslation?}&hq={hq?}&hl={hl?}&siteSearch={siteSearch?}&siteSearchFilter={siteSearchFilter?}&exactTerms={exactTerms?}&excludeTerms={excludeTerms?}&linkSite={linkSite?}&orTerms={orTerms?}&relatedSite={relatedSite?}&dateRestrict={dateRestrict?}&lowRange={lowRange?}&highRange={highRange?}&searchType={searchType}&fileType={fileType?}&rights={rights?}&imgSize={imgSize?}&imgType={imgType?}&imgColorType={imgColorType?}&imgDominantColor={imgDominantColor?}&alt=json"
  },
  "queries": {
    "request": [
      {
        "title": "Google Custom Search - Why always me?",
        "totalResults": "15070000000",
        "searchTerms": "Why always me?",
        "count": 5,
        "startIndex": 1,
        "inputEncoding": "utf8",
        "outputEncoding": "utf8",
        "safe": "off",
        "cx": "0361bdde161e94157"
      }
    ],
    "nextPage": [
      {
        "title": "Google Custom Search - Why always me?",
        "totalResults": "15070000000",
        "searchTerms": "Why always me?",
        "count": 5,
        "startIndex": 6,
        "inputEncoding": "utf8",
        "outputEncoding": "utf8",
        "safe": "off",
        "cx": "0361bdde161e94157"
      }
    ]
  },
  "context": {
    "title": "test1"
  },
  "searchInformation": {
    "searchTime": 0.326402,
    "formattedSearchTime": "0.33",
    "totalResults": "15070000000",
    "formattedTotalResults": "15,070,000,000"
  },
  "items": [
    {
      "kind": "customsearch#result",
      "title": "'Why Always Me?' - Balotelli's famous slogan meaning and history ...",
      "htmlTitle": "&#39;<b>Why Always Me</b>?&#39; - Balotelli&#39;s famous slogan meaning and history ...",
      "link": "https://www.goal.com/en-us/news/why-always-me-balotelli-famous-slogan-meaning-history-explained/bltb717f14c0dab3994",
      "displayLink": "www.goal.com",
      "snippet": "Feb 18, 2022 ... 'Why Always Me?' is a catchphrase coined by Mario Balotelli which was printed on an undershirt and revealed to the world as a message during the\u00a0...",
      "htmlSnippet": "Feb 18, 2022 <b>...</b> &#39;<b>Why Always Me</b>?&#39; is a catchphrase coined by Mario Balotelli which was printed on an undershirt and revealed to the world as a message during the&nbsp;...",
      "cacheId": "JfDuhxHb9Z4J",
      "formattedUrl": "https://www.goal.com/en-us/.../why-always-me.../bltb717f14c0dab3994",
      "htmlFormattedUrl": "https://www.goal.com/en-us/.../<b>why-always</b>-<b>me</b>.../bltb717f14c0dab3994",
      "pagemap": {
        "cse_thumbnail": [
          {
            "src": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQnyypYlojzXFoWzyH0ViYQvmcZekmFGfrzcp2QX_jIGF6logIDFeyHRC8",
            "width": "300",
            "height": "168"
          }
        ],
        "metatags": [
          {
            "og:image": "https://assets.goal.com/v3/assets/bltcc7a7ffd2fbf71f5/blt7b107b5325673a64/620e996c9049b9033b8cb542/goal---web-three-way-split-window--b6da7676-739f-40f3-827c-29e32c007096.jpeg",
            "og:type": "article",
            "og:image:alt": "Mario Balotelli Manchester City Manchester United 2011 Why Always Me?",
            "og:image:width": "1920",
            "twitter:card": "summary_large_image",
            "og:title": "\u2018Why Always Me?\u2019 - Balotelli\u2019s famous slogan meaning and history explained | Goal.com US",
            "og:image:height": "1080",
            "og:description": "Mario Balotelli's 'Why Always Me?' catchphrase explained, when it happened and everything you need to know",
            "twitter:image": "https://assets.goal.com/v3/assets/bltcc7a7ffd2fbf71f5/blt7b107b5325673a64/620e996c9049b9033b8cb542/goal---web-three-way-split-window--b6da7676-739f-40f3-827c-29e32c007096.jpeg",
            "next-head-count": "69",
            "twitter:image:alt": "Mario Balotelli Manchester City Manchester United 2011 Why Always Me?",
            "twitter:site": "@goal",
            "viewport": "width=device-width, initial-scale=1, maximum-scale=1",
            "twitter:description": "Mario Balotelli's 'Why Always Me?' catchphrase explained, when it happened and everything you need to know"
          }
        ],
        "cse_image": [
          {
            "src": "https://assets.goal.com/v3/assets/bltcc7a7ffd2fbf71f5/blt7b107b5325673a64/620e996c9049b9033b8cb542/goal---web-three-way-split-window--b6da7676-739f-40f3-827c-29e32c007096.jpeg"
          }
        ]
      }
    },
    ... (4 more results)    
  ]
}
{% endhighlight %}
The retrieved information lies within `items`, and one can retrieve their url with `results.json()['items'][item_index]['link']`.

**Web scraping result urls**

To further analyze the content of the results, one can use web scraping tools to extract the source html from the retrieved sites. In Python, for static sites, the `BeautifulSoup` library is good enough. However for dynamic sites (which is the majority of sites), we may also need the `Selenium` library to first run the hidden javascript in a headless browser, otherwise you might get near empty responses or something like `Please enable JS and disable any ad blocker.` A sample code snippet to retrieve the source html of static sites is:

{% highlight python %}
import requests
from bs4 import BeautifulSoup

url = <url_of_a_static_site>
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

for paragraph in soup.find_all("p"):
    print(paragraph.text)
{% endhighlight %}

A sample code snippet for dynamic sites is:
{% highlight python %}
from bs4 import BeautifulSoup
from selenium import webdriver
import time

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

url = <url_of_a_dynamic_site>
driver = webdriver.Chrome(options = chrome_options)
driver.get(url)
time.sleep(2)
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
driver.close()

for paragraph in soup.find_all("p"):
    print(paragraph.text)
{% endhighlight %}

Here, the above `chrome_options` arguments are only needed when using selenium on a no-monitor server environment such as Google Colab. Moreover, one might need to install the necessary packages as so:

{% highlight bash %}
pip install undetected-chromedriver
apt-get update
apt install chromium-chromedriver chromium-browser
{% endhighlight %}

Even with selenium, there might be another problem - server side sometimes have anti-bot detection and blocks your access. The library [undetected-chromedriver][undetected_chromedriver] is able to solve this problem. However, by the time of this post, even after a lot of tries I was not able to use this library on Google Colab due to the following error:

`WebDriverException: Message: unknown error: cannot connect to chrome at 127.0.0.1:45613
from chrome not reachable`

Hopefully it gets fixed in the future. 

[google_search_api_url]: https://developers.google.com/custom-search/v1/overview
[google_cloud_billing]: https://cloud.google.com/billing/docs/how-to/manage-billing-account
[programmable_search_engine]: https://programmablesearchengine.google.com/controlpanel/all
[undetected_chromedriver]: https://github.com/ultrafunkamsterdam/undetected-chromedriver
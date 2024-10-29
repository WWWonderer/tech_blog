---
title: "Using OpenAI API in compute nodes of Compute Canada"
date: 2024-10-26
categories: "software_engineering"
---

**Problem statement**

Compute Canada has 2 kinds of nodes: compute nodes and login nodes. Login nodes have internet access and are the first nodes you access when connecting to the cluster using ssh. They are shared among all users and are used to schedule tasks to be run on compute nodes. Compute nodes have resources exclusively allocated to a dedicated user (according to your specifications in login nodes), but they do not have internet access. With the prevalence of LLMs, a common use case is to use OpenAI API (or whatever other provider) in your experimental code. This requires internet connection to access the API and a compute node to access the GPU resources, which also prevents slowing down the login nodes for the other users. How can we make the compute nodes access the OpenAI server? 

**Proxy server**

An easy way to do this is to use a proxy server. A proxy server is a server application that acts as an intermediary between a client and another resources server. Instead of querying the resources server directly, the client queries the proxy server which in turn queries the resources server. In our case, since the compute nodes are connected to the login nodes, which connects to the wider internet, we need to set up a login node to be the proxy server for our OpenAI API requests. 

On Compute Canada, we can use the `ssh` command to achieve the above by typing the following:

```ssh -D [local_port] [login_node] -N -f```

Here, `-D` binds a local port on the compute node to a login node using the `SOCKS` protocol through `ssh` tunneling, `-N` is an option to not execute remote commands, and `-f` requests `ssh` to go to the background. A working example can be `ssh -D 9999 narval1 -N -f` assuming you are on the Narval cluster.

**OpenAI request through proxy server**

Here is a code snippet of using server proxy in OpenAI API requests on compute nodes, using `9999` as example port and `narval1` as example login node:

{% highlight python %}
import requests

# Set up the SOCKS5 proxy, which is used when you create a tunnel through ssh
proxies = {
    "http": "socks5h://localhost:9999",
    "https": "socks5h://localhost:9999"
}

# Set your API key
api_key = "<your api key>"

# Define a JSON header
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Define the data for the ChatCompletion request
data = {
    "model": "gpt-3.5-turbo",  # or whatever model you like
    "messages": [{"role": "user", "content": "Educate me about proxy servers."}],
    "max_tokens": 100
}

# Send the request through the SOCKS proxy
response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=data,
    proxies=proxies
)

# Print the response from the model
print(response.json())
{% endhighlight %}

This should print something like the following: 
```
{'id': 'chatcmpl-ANU3zJv3VSjGjV81TFJAJNynQDe2R', 'object': 'chat.completion', 'created': 1730160471, 'model': 'gpt-3.5-turbo-0125', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "A proxy server acts as an intermediary between a user's device and the internet. When a user accesses the internet through a proxy server, their requests are first sent to the proxy server, which then forwards the request to the internet on behalf of the user", 'refusal': None}, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 14, 'completion_tokens': 50, 'total_tokens': 64, 'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'system_fingerprint': None}
```

If you are using Langchain, they have a built-in support for proxies in their OpenAI wrapper:
{% highlight python %}
from langchain_openai import ChatOpenAI
import os
os.environ['OPENAI_API_KEY'] = "<your api key>"

llm = ChatOpenAI(model='gpt-4o-mini', openai_proxy="socks5://localhost:9999")
llm.invoke('Please educate me about proxy servers.')
{% endhighlight %}
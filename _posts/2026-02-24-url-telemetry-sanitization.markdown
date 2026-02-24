---
title: "URL Telemetry Sanitization"
date: 2026-02-24
categories: "software_engineering"
---

I have built an activity tracker that logs the active window URL for classification purposes. However, as I started storing these URLs, I became concerned about the potential security risks if the logs were ever leaked or stolen. URLs can sometimes contain sensitive information, and in the worst case, this could expose access to personal accounts. This concern led me down the rabbit hole of understanding URL structures and how to sanitize them safely when building the v2 tracker.

## URL anatomy

A typical URL looks like:

`scheme://host/path?query#fragment`

An example from my logged URLs (which is nothing but the nba standings table shown in Google):

`https://www.google.com/search?client=firefox-b-d&q=nba#sie=lg;/g/11m6vw9pbt;3;/m/05jvx;st;fp;1;;;`

The components are as follow:
- **scheme**: protocol (e.g., `https`). It tells the browser how to communicate with the server. For example, `https` means the connection is encrypted using TLS, while `http` is unencrypted.

- **host**: domain name (e.g., `www.google.com`). It identifies the server being accessed.

- **path**: resource location on the server (e.g., `/search`). It specifies which page or file is requested.

- **query**: key-value parameters after `?` and separated by `&` (e.g., `q=nba&client=firefox`). It is used to pass user input, session identifiers, or navigation state.

- **fragment**: client-side reference after `#` (e.g., `#sie=lg;/g/11m6vw9pbt;3;/m/05jvx;st;fp;1;;;`). It is not sent to the server and is instead processed by the browser and the page’s JavaScript. It is commonly used to store UI state, such as scroll position or selected content. For example, if I remove this fragment from the above Google search URL, Firefox no longer shows the NBA standings box and instead shows the default NBA games view. Also, this fragment appears to be browser-dependent: Safari does not show the standings box even when using the same fragment generated on Firefox.

In practice, the **query** (and sometimes the **fragment**) is where sensitive data most often appears.

## Nested URLs inside query parameters

Some sites (especially authentication flows) put an entire URL inside a query value. Because URLs cannot contain spaces and certain characters such as `/`, `?`, `&`, and `=` have reserved structural meaning, any URL embedded inside another URL must first be URL-encoded.

URL-encoding replaces spaces and special characters with `%` followed by their hexadecimal ASCII code. This ensures the embedded URL is treated as a single parameter value rather than being misinterpreted as part of the outer URL structure.

The table below shows some common percent-encoding examples:

| Encoded | Character | Purpose |
|--------|-----------|---------|
| `%20` | (space) | Spaces are not allowed in URLs |
| `%3A` | `:` | Separates the scheme (`https:`) |
| `%2F` | `/` | Path separator |
| `%3F` | `?` | Begins the query string |
| `%3D` | `=` | Separates parameter key and value |
| `%26` | `&` | Separates query parameters |
| `%23` | `#` | Begins the fragment |
| `%25` | `%` | Literal percent character |

For example, a login page may include a “return to” URL:

`https://www.amazon.ca/ap/signin?openid.return_to=https%3A%2F%2Fna.primevideo.com%2Fauth%2Freturn%2F...`

The important detail is that `openid.return_to=...` is **one** parameter. Its value is a URL-encoded URL. After decoding, the inner URL may itself contain its own query parameters like `_t%3D` (`_t=`) or `location%3D` (`location=`).

This is why you may see multiple `=` signs “inside” a single outer parameter after decoding: they belong to the **inner** URL’s query.

## Simple sanitization heuristics

The simplest sanitization is as follow:

- keep `scheme`, `host`, and `path`
- remove the entire `query` and `fragment`

Example:

Before: `https://example.com/page?token=secret&user=123#profile`  
After:  `https://example.com/page`

This scheme offers the maximum safety, but can also strip away useful non-sensitive details. For example, all authentication tokens will be (rightfully) removed, but so will be google search queries. For a classification task, it therefore does not offer enough precision for contents of the same website.

In fact, the pragmatic goal of URL telemetry sanitization is to achieve and balance the following two opposing objectives:
1. prevent account takeover or leaking secrets if logs ever escape.
2. keep enough signals for classification. 

The simplest sanitization scheme above achieves objective 1 completely, but doesn't do well for objective 2. We therefore design some custom heuristic rules below to improve on it:

**Rule 1: always drop fragments.**
Fragments (`#...`) are not sent to the server, and are mainly used for in-page navigation or client-side UI state. They do not offer much classification benefits but may contain sensitive information such as json web tokens (JWTs). 

**Rule 2: always drop userinfo (user:pass@host) before host.**
Certain older URLs can embed credentials like `http://user:pass@example.com/...`, we should remove the userinfo part of the URL, and this will not affect classification outcomes.

**Rule 3: always keep scheme + host.**
Scheme and host identify the website being accessed and are essential for classification. While they may reveal infrastructure details (e.g., `https://my-private-bucket.s3.amazonaws.com/`), they rarely contain authentication secrets and provide high classification value.

**Rule 4: sanitize path segments.**
Path segments can contain personally identifiable information (PII) (e.g., `/users/13579`, `profile/550e8400-e29b-41d4-a716-446655440000`). These information do not help classification and should be redacted. We can use regex rules to identify digit groups, UUID groups etc. in path segments, and replace them with placeholders. For example, `users/13579/orders/550e8400...` will be sanitized into `users/[INT]/orders/[UUID]`.

**Rule 5: always sanitize sensitive query keys.**
This is practically the most important part. In real URLs, sensitive information are often passed using a set of predefined keys, such as `token`, `session`, `password`, etc. For example, consider the following real log: `https://platform.openai.com/auth/callback?code=...&scope=...&state=...`, the `code` parameter is an OAuth authorization code that can be exchanged for access tokens, and the `state` parameter contains session-specific information. If a hacker somehow gets these information before the session expires, the account may be compromised. We should always sanitize query values of these sensitive keys. 

**Rule 6: sanitize sensitive values.**
Sensitive values can also hide in innocuous looking keys such as `q` or `data`, and sometimes in nested URLs, where even sensitive keys such as `token` will not be parsed as a key but part of a value. To make the sanitization more robust, we can use the same regex heuristics that redact sensitive path segments to redact sensitive values in the URL. Specifically, we redact values that look like structured identifiers or secrets. Anything that looks like a JWT, a UUID, a base64url, etc. needs to be redacted. We should also redact values by having a maximum length cap, as they are most likely autogenerated and do not involve useful classification information.

## Regex Implementations

Below are some regex implementations of the above heuristics. Each regex has a comment on its meaning and purpose. 

```python
import re

# UUID v4-like pattern: 8-4-4-4-12 hex digits (e.g., 550e8400-e29b-41d4-a716-446655440000)
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Long hex strings (often hashes, IDs, or tokens) (e.g., a3f1b9c4d8e7f123)
HEX_RE = re.compile(r"^[0-9a-fA-F]{16,}$")

# Base64url-ish tokens (often OAuth/JWT parts, IDs, or serialized state). Base64 is reversible, not encryption.
# (e.g., 4f3GhT-2Lk8vPq9sXzA1BcDeF)
B64URL_RE = re.compile(r"^[A-Za-z0-9_-]{24,}$")

# JWT-like structure: three base64url segments separated by dots (header.payload.signature)
# (e.g., eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.aaa.bbb)
JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")

# Email address in a URL segment or parameter value (e.g., johndoe@example.com)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
```

These heuristics can of course be improved, and can both result in false positive redactions and missing to redact short sensitive information. But generally, they are fairly robust and acceptable for the purpose of a classification logger. 



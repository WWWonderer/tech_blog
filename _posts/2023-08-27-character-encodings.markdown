---
title: "Character encodings"
date: 2023-08-27
categories: "information_retrieval"
---
**What are character encodings**

Modern computers are binary machines and store all information using bits. For numbers, this would be easy as we can simply store the binary value of a number: a digital '10' can be stored as a binary '1010' in memory (RAM or disk). Graphics can be quantified as numerical color maps (such as RGB map), and thus do not require new 'inventions' to be stored. However, letters, characters and other repeating symbols such as emojis do need to be first 'converted' to a number in an arbitrary way before being able to be stored. This is where character encodings come from. Since this process is arbitrary, many encodings exist, but a few of them became popular and formed a standard. 

**Terminology**

First, let's get more concrete with the terminology of character encodings. We have the following definitions:

_grapheme_: a single unit of a human writing system, which can be a letter such as 'd', a character such as '空', an emoji such as ':smile:', etc.

_code point_: code point is a unique position in a quantized character mapping scheme. For example, in 'ASCII', the code point associated with letter 'd' is the decimal value '100'. In more complex encoding schemes such as 'UTF-8', certain graphemes such as 'é' can be mapped in two ways: it has its own code point '233', or it can be represented as the code point of 'e' (101) combined with the code point of acute accent modifier (769).

_encoding_: encodings are the binary representations of the graphemes. Through an encoding map, each grapheme is linked to a corresponding code point, which can then be translated into an encoding. However, the size of the encoding is an arbitrary choice. For example, in 'UTF-32', every grapheme will be mapped into 4 bytes, even the simpler ones such as the letter 'd' (`d -> 100 -> 00000000 00000000 00000000 01100100`), yet in 'UTF-32', the simpler letters such as 'd' only take up 1 byte (`d -> 100 -> 01100100`). Different choices of encoding will thus affect memory consumption. 

All the above form the basic blocks of all encoding standards.

**ASCII**

ASCII (American Standard Code for Information Exchange) is an early and influential character encoding standard. It has only 128 code points, and only 95 out of the 128 are printable characters. ASCII is therefore limited in scope, however due to its simplicity, ASCII has the benefit that all graphemes has a unique 1 byte encoding. Thus, the lens of the encoding is the same as the lens of graphemes. Here is the full ASCII table:  

| Code point (Dec)  | Hex  | Oct  | Char                             |
|------|------|------|----------------------------------|
|   0  |  00  |  000 | NUL (null)                       |
|   1  |  01  |  001 | SOH (start of heading)           |
|   2  |  02  |  002 | STX (start of text)              |
|   3  |  03  |  003 | ETX (end of text)                |
|   4  |  04  |  004 | EOT (end of transmission)        |
|   5  |  05  |  005 | ENQ (enquiry)                    |
|   6  |  06  |  006 | ACK (acknowledge)                |
|   7  |  07  |  007 | BEL (bell)                       |
|   8  |  08  |  010 | BS  (backspace)                  |
|   9  |  09  |  011 | TAB (horizontal tab)             |
|   10  |  0A  |  012 | LF  (line feed)                  |
|   11  |  0B  |  013 | VT  (vertical tab)               |
|   12  |  0C  |  014 | FF  (form feed)                  |
|   13  |  0D  |  015 | CR  (carriage return)            |
|   14  |  0E  |  016 | SO  (shift out)                  |
|   15  |  0F  |  017 | SI  (shift in)                   |
|   16  |  10  |  020 | DLE (data link escape)           |
|   17  |  11  |  021 | DC1 (device control 1)           |
|   18  |  12  |  022 | DC2 (device control 2)           |
|   19  |  13  |  023 | DC3 (device control 3)           |
|   20  |  14  |  024 | DC4 (device control 4)           |
|   21  |  15  |  025 | NAK (negative acknowledgment)    |
|   22  |  16  |  026 | SYN (synchronous idle)           |
|   23  |  17  |  027 | ETB (end of transmission block)  |
|   24  |  18  |  030 | CAN (cancel)                     |
|   25  |  19  |  031 | EM  (end of medium)              |
|   26  |  1A  |  032 | SUB (substitute)                 |
|   27  |  1B  |  033 | ESC (escape)                     |
|   28  |  1C  |  034 | FS  (file separator)             |
|   29  |  1D  |  035 | GS  (group separator)            |
|   30  |  1E  |  036 | RS  (record separator)           |
|   31  |  1F  |  037 | US  (unit separator)             |
|   32  |  20  |  040 |     (space)                      |
|   33  |  21  |  041 | !                                |
|   34  |  22  |  042 | "                                |
|   35  |  23  |  043 | #                                |
|   36  |  24  |  044 | $                                |
|   37  |  25  |  045 | %                                |
|   38  |  26  |  046 | &                                |
|   39  |  27  |  047 | '                                |
|   40  |  28  |  050 | (                                |
|   41  |  29  |  051 | )                                |
|   42  |  2A  |  052 | *                                |
|   43  |  2B  |  053 | +                                |
|   44  |  2C  |  054 | ,                                |
|   45  |  2D  |  055 | -                                |
|   46  |  2E  |  056 | .                                |
|   47  |  2F  |  057 | /                                |
|   48  |  30  |  060 | 0                                |
|   49  |  31  |  061 | 1                                |
|   50  |  32  |  062 | 2                                |
|   51  |  33  |  063 | 3                                |
|   52  |  34  |  064 | 4                                |
|   53  |  35  |  065 | 5                                |
|   54  |  36  |  066 | 6                                |
|   55  |  37  |  067 | 7                                |
|   56  |  38  |  070 | 8                                |
|   57  |  39  |  071 | 9                                |
|   58  |  3A  |  072 | :                                |
|   59  |  3B  |  073 | ;                                |
|   60  |  3C  |  074 | <                                |
|   61  |  3D  |  075 | =                                |
|   62  |  3E  |  076 | >                                |
|   63  |  3F  |  077 | ?                                |
|   64  |  40  |  100 | @                                |
|   65  |  41  |  101 | A                                |
|   66  |  42  |  102 | B                                |
|   67  |  43  |  103 | C                                |
|   68  |  44  |  104 | D                                |
|   69  |  45  |  105 | E                                |
|   70  |  46  |  106 | F                                |
|   71  |  47  |  107 | G                                |
|   72  |  48  |  110 | H                                |
|   73  |  49  |  111 | I                                |
|   74  |  4A  |  112 | J                                |
|   75  |  4B  |  113 | K                                |
|   76  |  4C  |  114 | L                                |
|   77  |  4D  |  115 | M                                |
|   78  |  4E  |  116 | N                                |
|   79  |  4F  |  117 | O                                |
|   80  |  50  |  120 | P                                |
|   81  |  51  |  121 | Q                                |
|   82  |  52  |  122 | R                                |
|   83  |  53  |  123 | S                                |
|   84  |  54  |  124 | T                                |
|   85  |  55  |  125 | U                                |
|   86  |  56  |  126 | V                                |
|   87  |  57  |  127 | W                                |
|   88  |  58  |  130 | X                                |
|   89  |  59  |  131 | Y                                |
|   90  |  5A  |  132 | Z                                |
|   91  |  5B  |  133 | [                                |
|   92  |  5C  |  134 | \                                |
|   93  |  5D  |  135 | ]                                |
|   94  |  5E  |  136 | ^                                |
|   95  |  5F  |  137 | _                                |
|   96  |  60  |  140 | `                                |
|   97  |  61  |  141 | a                                |
|   98  |  62  |  142 | b                                |
|   99  |  63  |  143 | c                                |
|   100  |  64  |  144 | d                                |
|   101  |  65  |  145 | e                                |
|   102  |  66  |  146 | f                                |
|   103  |  67  |  147 | g                                |
|   104  |  68  |  150 | h                                |
|   105  |  69  |  151 | i                                |
|   106  |  6A  |  152 | j                                |
|   107  |  6B  |  153 | k                                |
|   108  |  6C  |  154 | l                                |
|   109  |  6D  |  155 | m                                |
|   110  |  6E  |  156 | n                                |
|   111  |  6F  |  157 | o                                |
|   112  |  70  |  160 | p                                |
|   113  |  71  |  161 | q                                |
|   114  |  72  |  162 | r                                |
|   115  |  73  |  163 | s                                |
|   116  |  74  |  164 | t                                |
|   117  |  75  |  165 | u                                |
|   118  |  76  |  166 | v                                |
|   119  |  77  |  167 | w                                |
|   120  |  78  |  170 | x                                |
|   121  |  79  |  171 | y                                |
|   122  |  7A  |  172 | z                                |
|   123  |  7B  |  173 | &#123;                                |
|   124  |  7C  |  174 | &#124;                                |
|   125  |  7D  |  175 | &#125;                                |
|   126  |  7E  |  176 | ~                                |
|   127  |  7F  |  177 | DEL (delete)                     |


**Unicode**

As ASCII is too limited in scope and does not include languages such as Chinese, Arabic, emojis, etc. A standard known as Unicode is formed to consistently handle text across most of the world's writing systems. Unicode is synchronized with international standard setting bodies such as ISO and is labeled ISO/IEC 10646. Due to ASCII's influence on early computing, Unicode is backward compatible with ASCII, maintaining the same code points and characters in its mapping scheme. However, different encoding schemes for Unicode exist, the most popular one being 'UTF-8'. Others include 'UTF-16', 'GB 18030' (Chinese), etc.

Since unicode comprises of text all across the world, it is organized in blocks, where each block is a continuous range of similar character codes. For example, the block `Basic Latin` comprises the first 128 character codes of unicode, and is nothing but the original ASCII encodings. The block `Cyrillic` comprises of the 1024th to 1280th character codes, and incorporate the basics of the Cyrillic languages. There are 327 blocks in total.

**UTF-8**

UTF-8 is the most popular encoding scheme of Unicode, unlike UTF-32 for example, UTF-8 has varying encoding sizes for different lexemes. The more common letters such as those in ASCII only require 1 byte of memory, while the more complex graphemes such as ':monkey:' may take up to 4 bytes of memory.

In order to determine the number of bytes needed during decoding, UTF-8 encoding's first byte has special meaning, below are the 4 forms of the first byte:

```
0xxxxxx -> 1 byte encoding 
110xxxxx -> 2 bytes encoding
1110xxxx -> 3 bytes encoding
11110xxx -> 4 bytes encoding
```

bytes 2 to 4 only start with `10xxxxxx`, thus it is possible to determine when a new grapheme has started. For ASCII (or Basic Latin) code points, UTF-8 and ASCII result in the same encodings. 

**Mojibake**

Mojibake is the garbled and unintelligible text that is the result of text being decoded using an unintended character decoding. This display may include the generic replacement character ("�") in places where the binary representation is considered invalid. Opening a binary file with a text editor for example will result in mojibake, as the bytes in memory are not intended to be decoded as text at all. 

Mojibake can also happen when decoding text using a different character encoding scheme than what was used in encoding the text. This is especially likely to happen when the encoding is done using multi-bytes schemes such as UTF-8, but decoded using 1 byte schemes such as ASCII. For example, when we try to decode :see_no_evil:, which has a four bytes hexadecimal encoding of `0xF0 0x9F 0x99 0x88`, but instead uses a 1 byte decoding scheme, we will decode `0xF0` into the mojibake `ð`, `0x9F, 0x99, 0x88` into the mojibakes `���`. Modern browsers do a great job at guessing the encodings used by the websites and thus decode accordingly. However, manually determining a webpage's decoding such as when web scraping can be a troublesome process. 

**Other encoding schemes**

Some other encoding schemes occasionally pop up. One is called ISO-8859-1 which is a 1 byte encoding scheme for extended Latin alphabets. It precedes Unicode and heavily influences it, thus is compatible with Unicode for much of its code points. Windows-1252 is a similar encoding for Latin alphabets proposed by Microsoft and used by default in Microsoft Windows. It is similar to ISO-8859-1 for most of its code points, and at the time of this writing is the most-used single byte encoding scheme in the world. Browsers treat ISO-8859-1 and ASCII as Windows-1252.
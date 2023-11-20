---
title: "VSCode path bug"
date: 2023-11-16
categories: "software_engineering"
---

**Why is my relative path broken in VSCode debugger and cannot be fixed?**

This topic probably does not deserve a separate post, but it is so annoying, and I *have to* pinpoint it. Below I will break down this seemingly simple problem, till it is engrained in my brain and the solution pops out automatically when I smell its trace. What a time waster! Hopefully others with similar experiences as me can do the same.

**Where does VSCode's *Debug Button* run the scripts from?**

When we click on the *Debug Python File* button on the top right of the VSCode interface, the latter spawns a separate shell terminal and launches the debugger. The general process is as follow:

```
cd <path/to/cwd>; <path/to/(venv)/python> <path/to/debugger> <port> -- <path/to/script.py>
```

Here, `cwd` signifies the current working directory, and any relative path in your python scripts are relative to this `cwd`. This makes where to run the script important as it affects whether your relative paths within your script are correct. The `cwd` along with other settings can be modified within the automatically created `launch.json` file. The latter typically resides within the `.vscode` folder directly under the root project folder. By default, VSCode sets the `cwd` to `${WorkspaceFolder}`, which is the root of your project folder, or the top folder in the explorer. Thus, if you have written relative paths relative to the script itself and not the root folder, the execution may lead to path errors. We can fix this by adjusting the `cwd` within `launch.json`. **However, the adjusted `cwd` will not be recognized if you click on the *Debug Python File* button, and will only be recognized if you run the debugger through `File -> Run -> Start Debugging` or by pressing *F5*.** [This is a known issue since forever, but remains unfixed](https://github.com/microsoft/vscode-cpptools/issues/8084).

**Solution/Workaround**

One can work around this problem in Python by using the following code:

{% highlight python %}
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, "<next folder in hierarchy>", ..., "<destination folder/file>")
{% endhighlight %}
This way, the script's relative path will be independent of the current working directory, and can hopefully save some debugging pain in the end. 





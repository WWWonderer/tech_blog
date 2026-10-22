---
title: 'Revisiting print("hello world")'
date: 2026-10-22
categories: "software_engineering"
---

Late one night, while thinking about Docker and containerization, I went into a rabbit hole of program execution pipelines. I want to record this journey by re-tracing the execution of a simple `hello world` program.

In almost every introductory programming class, students are asked to write something like:

```python
print("hello world")
```

The instructor will ask you to press Enter, then say something like:
`Congratulations! You've written your first program.`

I remember finding this rather underwhelming. I typed a few words, pressed Enter, and the computer repeated those words back to me. What was so special about that?

Only after studying the field for many years and after starting to contribute in the industry, I am beginning to appreciate that perhaps the most remarkable thing about this little program is precisely how unremarkable it feels. Enormous engineering efforts and ingenuity are poured into its mundane execution, let's unpack them below.

#### 1. **What happens when a line of code is being executed**

Consider the same program again:

```python
print("hello world")
```

At first glance, Python simply reads the instruction and displays the string.

But Python is itself a program. Something must execute Python before Python can execute our code. What about the thing that executes the something that executes Python? In the end, we are talking to metals, rubbers and plastics in the name of hardware. Then how do we standardize it such that my `hello world` program executes the same on a different hardware such as your machine, a server, or even a virtual machine or a phone? What makes this little `hello world` on the instructor's Mac computer also work on a student's Windows machine? When it comes to details, the questions seem endless, but we will think of a general framework in the following graph:

[![Hello world program execution]({{ site.baseurl }}/assets/images/p31_hello_world_1.png){:style="display:block; margin-left:auto; margin-right:auto"}]({{ site.baseurl }}/assets/images/p31_hello_world_1.png "Open full-size image"){:target="_blank" rel="noopener"}

The above graph is simplified, as each listed element can be further expanded with more elements and relationships, but in general, by standardizing these layers (especially the interpreter, OS, and HAL), our little `hello world` program can run the same on different hardware: a student Windows machine, the instructor's Mac, a Linux server, or even a virtual machine. The same python code, with the same interpreter, works across platforms because **each layer above the hardware provides a consistent interface**.

Now back to python (other languages might not need the interpreter). In a typical CPython implementation, our source code is compiled into Python bytecode, which is then executed by the Python interpreter. The interpreter itself consists largely of compiled native machine instructions, which can be mapped and executed directly by the CPU during run time. We can expand the first 2 layers further and draw the following graph:

[![Hello world Python execution]({{ site.baseurl }}/assets/images/p31_hello_world_2.png){:style="display:block; margin-left:auto; margin-right:auto"}]({{ site.baseurl }}/assets/images/p31_hello_world_2.png "Open full-size image"){:target="_blank" rel="noopener"}

Step 1 in the diagram above takes place when we download and install Python on our computer, a process usually covered briefly by instructors with little explanation. However, this step is essential, as it prepares our computer to execute Python programs by installing the necessary components, including the Python runtime. Once installed, we can invoke these components to execute our Python code in Step 2.

Step 2 then invokes the Python executable (made available after Step 1) when we execute our Python program. This executable contains a compiler that first converts our source code, written according to Python syntax, into Python bytecode, as well as an interpreter that repeatedly reads and executes these bytecode instructions using native machine code.

But executing Python bytecode is only part of the story. Some operations require interacting with resources managed by the operating system. Consider our simple `print("hello world")` example. When the interpreter executes the corresponding bytecode instructions, it eventually invokes Python's built-in `print()` function, which converts the supplied object into text and writes it to the standard output stream, usually `sys.stdout`.

If this stream is connected to a terminal, Python relies on operating-system facilities to deliver the output, typically through system calls made indirectly via system libraries. The terminal application then receives and interprets the output, rendering the corresponding characters on our screen.

This interaction also helps explain why Python provides different distributions for macOS, Windows, and Linux. Although the Python language remains largely the same across platforms, the underlying operating-system interfaces, executable formats, and hardware architectures can differ. Therefore, the Python runtime must be built to work with its target environment.


#### 2. **The operating system: the invisible coordinator**

In the previous section, we followed our little `print("hello world")` program from Python source code to bytecode, and eventually to native machine instructions executed by the CPU. We also saw that certain operations, such as writing to the terminal, require assistance from the operating system.

But this raises another question: **what exactly does the operating system provide to our Python program, and why do we need it in the first place?**

What follows might bring back some dreadful memories of operating systems classes, but broadly speaking, the OS is responsible for process scheduling, memory management, providing system-call interfaces, and managing access to hardware and other shared resources.

Consider the graph below:

[![Operating system services]({{ site.baseurl }}/assets/images/p31_hello_world_3.png){:style="display:block; margin-left:auto; margin-right:auto"}]({{ site.baseurl }}/assets/images/p31_hello_world_3.png "Open full-size image"){:target="_blank" rel="noopener"}

**Process scheduling**

Let's say we execute our little `hello world` program again.

As discussed in Section 1, our Python source code is compiled into bytecode and executed by the Python interpreter. But before that can happen, the interpreter itself must be launched.

When we enter `python hello.py` in a terminal, the shell requests the operating system to start a process running the Python executable.

Notice that **a program and a process are not the same thing**. A program is a collection of instructions, while a process is a running instance of a program, with its own execution state and resources. We can launch the same Python program multiple times, creating multiple independent processes.

Of course, our Python process is unlikely to be the only one running. We might have a browser open, music playing in the background, and numerous other processes performing various tasks.

The OS scheduler coordinates CPU execution among runnable threads. On a single CPU core, it can rapidly switch between threads by saving one thread's execution state and restoring another's. On multicore systems, multiple threads can execute simultaneously.

This allows many applications to make progress concurrently, giving us the impression that everything is running at once.

**Memory management**

Now consider our Python statement again:

```python
print("hello world")
```

The Python interpreter needs memory to store its own execution state, Python objects, and other runtime data.

When the Python process starts, the OS establishes its virtual address space, providing an isolated memory environment in which the interpreter can operate.

However, this doesn't mean the OS must allocate memory every time Python creates an object.

CPython has its own memory allocators that manage memory for Python objects. These allocators can reuse memory already available within the process, requesting additional memory from lower-level allocators when necessary. Those allocators may eventually request more memory from the OS.

For example, consider:

```python
objects = []

for _ in range(10000):
    objects.append(bytearray(b"hello world"))
```

Unlike our original example, this program repeatedly creates new objects and retains them in a list. As memory consumption grows, the Python runtime may eventually need additional memory from the OS.

The important distinction is that **the OS manages the process's virtual memory environment, while the Python runtime manages individual Python objects within that environment**.

Furthermore, different processes generally have separate virtual address spaces. Under normal circumstances, one process cannot directly access another process's private memory.

**System calls**

Now back to our `print("hello world")` example. After the Python interpreter executes the corresponding bytecode, it invokes the built-in `print()` function, which writes `"hello world\n"` to the standard output stream (`sys.stdout`).

When connected to a terminal, the newline typically causes Python to flush its output buffer, eventually invoking an OS system call such as `write()` on Linux. The kernel then delivers the bytes through the terminal interface, allowing the terminal application to display `hello world` on our screen.

This is an example of a **system call**, through which a process requests services from the OS kernel rather than accessing protected resources directly—in this case, the terminal interface. Other OS services, such as file I/O, network communication, and virtual memory allocation, are also accessible through system calls.

**Shared resources and isolation**

Putting these pieces together, the operating system provides an environment in which multiple processes can execute concurrently, use memory, and access shared resources in a controlled manner.

But while processes have separate execution states and generally isolated memory, they are not completely isolated from one another.

On Linux, ordinary processes often share common views of system resources, including process identifiers, filesystem mounts, and network interfaces. They may also share access to files and devices, subject to permissions.

The Linux kernel already provides mechanisms for controlling these boundaries. In particular, **namespaces** can give processes different views of system resources, while **control groups (cgroups)** can account for and limit the resources consumed by groups of processes.

This raises an interesting question: what if we deliberately grouped several processes together, gave them their own view of the system, and controlled how much CPU and memory they could consume?

We wouldn't necessarily need another operating system to accomplish this.

And that brings us to containers and Docker.

#### 3. **Docker: changing the environment without changing the kernel**

This was the question that originally sent me down this rabbit hole.

It is common practice in the industry to containerize applications, packaging them into seemingly self-contained units that can execute across different environments. Take our `hello world` example again. Let's put it into `hello.py`:

```python
print("hello world")
```

Then create a `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY hello.py .

CMD ["python", "hello.py"]
```

Build the image and run the container:

```bash
docker build -t hello-world-python .
docker run --rm hello-world-python
```

And we receive the familiar output:

```text
hello world
```

We often assume that this containerized application can now run on a server or someone else's machine, regardless of their local Python installation. But what exactly has Docker changed in the execution process?

Under the hood, Docker actually still relies on existing operating-system mechanisms. It does not create a new execution environment from scratch, but limits access and creates isolation on the existing environment. On Linux, containers are isolated groups of processes that share the host's Linux kernel, using features such as **namespaces** and **cgroups**.

- **Namespaces** provide isolated views of system resources, such as process IDs, network interfaces, and filesystem mounts.
- **Cgroups** manage and account for resource usage, including CPU and memory.

When we specify `FROM python:3.12-slim` in the Dockerfile, Docker uses a prebuilt image containing a compiled Python interpreter and its dependencies, essentially providing the Python installation discussed in Section 1. When we run the container, Docker creates an isolated execution environment using mechanisms such as namespaces and cgroups. Our `hello.py` program, which was copied into the image, is then executed by the Python interpreter within this isolated environment. The Python interpreter inside the container therefore is not aware of other system processes due to PID restrictions of the namespace. On the other hand, as the container leverages the host kernel, the host is aware of the processes within the container, and treat them just as normal processes.


The following diagram compares ordinary Linux processes with containerized processes:

[![Native Linux and Docker containerized execution compared]({{ site.baseurl }}/assets/images/p31_hello_world_4.png){:style="display:block; margin-left:auto; margin-right:auto"}]({{ site.baseurl }}/assets/images/p31_hello_world_4.png "Open full-size diagram"){:target="_blank" rel="noopener"}

Notice that both systems use the same Linux kernel, and the fundamental execution path remains unchanged: our Python program is executed by the CPython interpreter, which ultimately executes native machine instructions and makes system calls when operating-system services are needed. It is totally possible that a process within container 1 is being context switched to another process within container 2. From the kernel perspective, there is nothing special about container processes.

This brings up another question: **if containers share the host kernel, why do Docker images often include Ubuntu or another Linux distribution?**

A Linux distribution consists of more than its kernel. It also provides user-space components such as system libraries, utilities, shells, and package managers. When we write `FROM ubuntu:24.04`, we are including Ubuntu's user-space filesystem, not installing another Linux kernel.

Similarly, our `python:3.12-slim` image already includes Python and a minimal Debian-based user-space environment. This environment provides the executable files, libraries, and dependencies required by our application, organized in a familiar Linux filesystem structure. It also includes utilities that allow us to interact with the environment through a command-line interface, while the actual execution of processes remains the responsibility of the shared Linux kernel.

So Docker does not eliminate operating-system dependencies. Rather, it standardizes the application's user-space environment while relying on the underlying kernel to execute processes and provide operating-system services.

Our little `hello world` program still follows essentially the same execution path. Docker changes the environment surrounding that execution, not the fundamental mechanism by which Python instructions eventually run on the CPU.


#### 4. **Beneath the software, another universe**

We started with a simple `print("hello world")`, but following its execution led us through interpreters, operating systems, kernels, containers, and eventually the hardware itself.

And yet, we've barely scratched the surface. Beneath each abstraction lies another world of complexity, built upon decades of research and engineering. On the hardware side, which I know almost nothing about, the historical progress described by the famous Moore's law was made possible through enormous and sustained efforts by semiconductor engineers in transistor design and chip manufacturing.

Zooming out, in today's AI era, similarly tremendous efforts are being made by GPU manufacturers and data center engineers to scale computing infrastructure, alongside countless low-level software engineers developing the frameworks that make large-scale AI systems possible. What once required enormous technical expertise is gradually becoming accessible through increasingly simple interfaces, even as the systems underneath grow more complex.

Perhaps one of the greatest achievements of computer science is not simply making extraordinary things possible, but making them ordinary. We write a line of Python, run it on another machine, or package it into a container, rarely thinking about the enormous machinery that makes it all work. Perhaps soon, we will routinely command entire networks of AI agents to execute tasks for us through voice.

Eventually, an introductory programming exercise might involve designing a simple workflow and telling an agent to execute it.

But behind every seemingly simple command, we should remember **the extraordinary achievements and countless contributions of the engineers at every level who made it all possible.**


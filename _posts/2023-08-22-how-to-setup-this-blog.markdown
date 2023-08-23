---
title: "How to setup this blog"
date: 2023-08-22
categories: "software_engineering"
---

**Github Pages and Jekyll**

This blog is published using [GitHub Pages][github-pages], which in turn uses [Jekyll][jekyll] by default to build and deploy the site. GitHub Pages is a free hosting service offered by GitHub, where a GitHub repository serves as the website root. Jekyll is a static site generator written in Ruby, which is able to transform markup languages into a static website. Once GitHub Pages is enabled for a repository, GitHub uses Jekyll in the backend to build and deploy your website based on the content of the repository.

**How does Jekyll build a site**

The commands `mkdir <site-repo> && cd <site-repo> && jekyll new .` can be used to initialize the necessary structures of a jekyll website as follow:

```
-<site-repo>
  -_posts
    -some_post.markdown
  -Gemfile
  -Gemfile.lock
  -_config.yml
  -about.markdown
  -index.markdown
```

Then, `jekyll serve` will build and deploy your website at port 4000 of localhost. The deployed website looks similar to the current blog, and posts inside the `_posts` directory will be correctly displayed as blog posts. However, the site repository is relatively simple and does not contain any formatting information, how does jekyll build a good looking site from so little?

**Jekyll themes**

The trick lies in the `_config.yml` file and the line `theme: minima`. In fact, as per the official documentation, "Jekyll has an extensive theme system that allows you to leverage community-maintained templates and styles to customize your site’s presentation. Jekyll themes specify plugins and package up assets, layouts, includes, and stylesheets in a way that can be overridden by your site’s content". 

The reason why the default site is looking good is because Jekyll by default installs and builds from a theme called `minima`, which is downloaded as part of `Gemfile` and provides the necessary html templates for the site's presentation. The path to these templates can be accessed using `bundle info --path minima`, which in my Windows computer returns `C:/Ruby32-x64/lib/ruby/gems/3.2.0/gems/minima-2.5.1`. When opening it, we see the following structure:
```
-minima-2.5.1
  -_includes
    -header.html
    -footer.html
    -...
  _layouts
    -home.html
    -post.html
    ...
  _sass
  -assets
  -LICENSE
  -README
```
Jekyll in the background takes these files in conjunction to our own files in `<site-repo>` to create our site, and **by copying and modifying the files inside the `_includes` folders etc. to our own `<site-repo>`, we are able to overwrite the default theme and customise our own site presentation.** 

**Jekyll blogs**

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. 



After that, we can include the necessary front matter. A front matter is written in `YAML` format and must be the first thing in the file between triple-dashed lines. An example is as follow:

```
---
title: what is a title?
date: 2023-08-22
---
```
The variables defined in the front matter will be available to use by you using `Liquid` tags, and some of them will be used by Jekyll. For example, the current url of `https://wwwonderer.github.io/tech_blog/software_engineering/2023/08/22/how-to-setup-this-blog.html` uses the `categories` (/software_engineering/) and `date` variables of the front matter.

**Local Deployment**

As discussed above, GitHub uses Jekyll to build and deploy the site on their backend. We can choose a branch and by default any commit and push to that branch will trigger an autodeployment on the GitHub server. However, to be able to preview what we push, we need local deployment using Jekyll to replicate what GitHub does. I found 2 ways of doing so:

* **Install Ruby etc. to be able to run Jekyll locally**

  On Windows, Ruby and its associated package/environment managers (gem, bundler) can be installed using [RubyInstaller][rubyinstaller]. 
  
  Once done, we can install Jekyll with:
  ```
  gem install jekyll
  ```

  For Ruby version3.0.0 or higher, we need to add `WEBrick` to our dependencies:

  ```
  bundle add webrick
  ```

  Finally, navigate to `<site_repo>` and do:

  ```
  bundle exec jekyll serve
  ```
  to host the site at localhost:4000. 

* **Using [Docker][docker]**

  Even though Ruby might be extremely cool, not everyone is familiar with it (including myself), and not everyone wants to install yet another language and everything associated with it to their computer for just one simple application. Docker solves this problem by providing the necessary environment for Jekyll to be able to edit and preview your blog. 

  First, we can get a Docker image of Jekyll:

  ```
  docker pull jekyll/jekyll
  ```

  Then, we can navigate to our `<site-repo>` and run the following command:

  ```
  docker run -it --rm -v ".:/srv/jekyll" -p "4000:4000" jekyll/jekyll jekyll serve --force_polling
  ```

  This command will establish an environment that includes Ruby and Jekyll from Docker, mount the current directory to it and forward port 4000 of this environment to ours. We can thus still test our blog without the bigger overhead of installing Ruby and its associated gems.

[jekyll]: https://jekyllrb.com/
[github-pages]: https://pages.github.com/
[docker]: https://www.docker.com/
[rubyinstaller]: https://rubyinstaller.org/
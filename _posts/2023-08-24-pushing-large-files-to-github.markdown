---
title: "Pushing large files to GitHub"
date: 2023-08-24
categories: "software_engineering"
---
**Pushing large files to GitHub**

When pushing large files (file size > 100MB = GitHub file size limit) to GitHub, one might experience errors such as below:

```
error: RPC failed; HTTP 408 curl 22 The requested URL returned
... 
fatal: the remote end hung up unexpectedly
```

Here, `RPC` stands for "Remote Procedure Call" between your local repository and the GitHub repository, `HTTP 408` stands for request timeout. Many other error messages may pop up, but the culprit is usually that you are trying to push a large file over the internet.

**Increase postbuffer size**

One potential solution to the above problem when the large file is still relatively small (< 100M) is to increase the postbuffer size with:

```
git config --global http.postbuffer 500M
```
where `500M` or `<some other number>` is the new buffer size (the default size is `1M`). However, this solution is not generally effective, as per the [official documentation][postbuffer_doc], "raising this limit is only effective for disabling chunked
transfer encoding" (which happens over http/1.1 when buffer size is surpassed) and "therefore should be used only where the remote
server or a proxy only supports HTTP/1.0 or is noncompliant with the
HTTP standard".

To check existing postbuffer size, we can use:

```
git config --get http.postbuffer
```

**Git LFS**

The better solution is to use [Git Large File Storage][git_lfs] (LFS), which is designed to handle large files such as audio/video samples, datasets, etc. and replace them with text pointers inside Git, while storing them separately elsewhere.

To use LFS, one should install it and then specifically track large files with:

```
git lfs track "*.<extension>"
```

Afterwards, all files ending with `<extension>` will be replaced by a text pointer, and a `.gitattributes` file specifying the tracked extensions will be created. The original large files will be moved to the `lfs` cache inside the `.git` repository, and only their pointers will be tracked by git. You can then commit and push as usual. To view the files tracked by LFS, type:

```
git lfs ls-files --all --long
```

One caveat of the above is that `git lfs track` creates a new commit. Say you have `commit1` where you committed some large files which cannot be pushed to the remote repository, so you did `commit2` using LFS, and you push again. This will still fail as git pushes the commits sequentially, and while `commit2` adds the large files to LFS, `commit1` didn't and you get the same error. You can solve this error by squashing the last 2 commits as 1 using interactive rebase:

```
git rebase -i HEAD~2
```

edit rebase list in the interactive rebase editor (1st editor):

```
pick <commit1_hash> <Commit message for commit 1>
squash <commit2_hash> <Commit message for commit 2>
```

Optionally modify commit message in the commit message editor (2nd editor), and force push to remote with a new squashed history:

```
git push origin <branch_name> --force
```

To avoid the above, we can use the [migrate][git_lfs_migrate] feature of LFS with:

```
git lfs migrate import --include="*.<extension>* --everything
```

Here, LFS will track large `<extension>` files as well as rewrite history wherever they are involved, so no new commit is needed, but a forced push is required. To have an idea of what files are large, we can do:

```
git lfs migrate info
```

This command will print the extensions of the largest files in descending order of file size.

As a last note, one should be careful with the use of LFS and GitHub as GitHub only allows 1GB of free LFS quota, any more would need the purchase of additional data packs, where one pack costs $5 per month and provides 50GB of storage. If your data quota is exceeded, LFS will be disabled for your account.



[postbuffer_doc]: https://git-scm.com/docs/git-config#Documentation/git-config.txt-httppostBuffer
[git_lfs]: https://git-lfs.com/
[git_lfs_migrate]: https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-migrate.adoc



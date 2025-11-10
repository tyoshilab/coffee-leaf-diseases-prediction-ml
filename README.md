## coffee-leaf-diseases-prediction-ml — Quick pull instructions (Windows / macOS / Linux)

This README explains how to clone/pull this repository and obtain the dataset files
`dataset/test_images_224.npy` and `dataset/train_images_224.npy` on Windows, macOS and Linux.

---

## Repository

Clone the repo first (replace URL if you use SSH):

```bash
git clone https://github.com/tyoshilab/coffee-leaf-deseases-prediction-ml.git
cd coffee-leaf-diseases-prediction-ml
```

---

## Use Git LFS for `.npy` files

Prerequisite: install Git and Git LFS on your machine (instructions below per OS).

After cloning, fetch LFS objects and check out the files:

```bash
# ensure git-lfs is initialized for this user (one-time per machine)
git lfs install

# fetch LFS objects for the checked-out commits
git lfs pull

# now dataset files like dataset/test_images_224.npy should be present
ls -lh dataset/test_images_224.npy dataset/train_images_224.npy
```

If you get pointer files (text files that start with "version https://git-lfs.github.com/spec/v1"), then LFS objects weren't downloaded — run `git lfs pull` again or see troubleshooting below.

### Install Git LFS (per-OS)

Linux (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install git-lfs -y
git lfs install
git lfs version
```

If your distro's package is old, use the official packagecloud script:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt install git-lfs -y
git lfs install
```

RHEL/CentOS (yum/dnf):

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash
sudo yum install git-lfs -y    # or dnf on newer systems
git lfs install
```

macOS (Homebrew):

```bash
brew install git-lfs
git lfs install
git lfs version
```

Windows (PowerShell) — recommended installers:

Using winget (Windows 10/11):

```powershell
winget install --id Git.GitLFS -e --source winget
git lfs install
git lfs version
```

Or use the Git LFS Windows installer from https://github.com/git-lfs/git-lfs/releases and run the installer.

---

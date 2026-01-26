# evo-python-sdk

[GitHub Repository](https://github.com/SeequentEvo/evo-python-sdk)

## Prerequisites

Before you get started, make sure you have:

* **A registered Evo app**

    *Evo apps* provide the credentials necessary to generate Evo access tokens, which in turn provide access to your Evo data. An app can be created by you or by a member of your team.
    
    Register an Evo app in the [Bentley Developer Portal](https://developer.bentley.com/my-apps). For in-depth instructions, follow this [guide](https://developer.seequent.com/docs/guides/getting-started/apps-and-tokens) on the Seequent Developer Portal.

    NOTE: You must have a **Bentley developer account** in order to create apps. If you try to register an app using the link above but find that you don't have permission, contact your account administrator to get access.

* **A local copy of this repository**

    Clone the repository using Git or download it as a ZIP file from the green **Code** button at the top of the page.

* **A Python code editor, eg. VS Code, PyCharm**
    
    For running and editing the sample notebooks and other source code files.

## Getting started with Evo SDK development

This repository contains a number of sub-packages. You may choose to install the `evo-sdk` package, which includes all
sub-packages and optional dependencies (e.g. Jupyter notebook support), or choose a specific package to install:

| Package | Version | Description |
| --- | --- | --- |
| evo-sdk | <a href="https://pypi.org/project/evo-sdk/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk" /></a> | A metapackage that installs all available Seequent Evo SDKs, including Jupyter notebook examples. |
| evo-sdk-common | <a href="https://pypi.org/project/evo-sdk-common/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-sdk-common" /></a> | A shared library that provides common functionality for integrating with Seequent's client SDKs. |
| [evo-files](evo-python-sdk/evo-files) | <a href="https://pypi.org/project/evo-files/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-files" /></a> | A service client for interacting with the Evo File API. |
| [evo-objects](evo-python-sdk/evo-objects) | <a href="https://pypi.org/project/evo-objects/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-objects" /></a> | A geoscience object service client library designed to help get up and running with the Geoscience Object API. |
| [evo-colormaps](evo-python-sdk/evo-colormaps)  | <a href="https://pypi.org/project/evo-colormaps/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-colormaps" /></a> | A service client to create colour mappings and associate them to geoscience data with the Colormap API.|
| [evo-blockmodels](evo-python-sdk/evo-blockmodels) | <a href="https://pypi.org/project/evo-blockmodels/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-blockmodels" /></a> | The Block Model API provides the ability to manage and report on block models in your Evo workspaces. |
| [evo-compute](evo-python-sdk/evo-compute)  | <a href="https://pypi.org/project/evo-compute/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/evo-compute" /></a> | A service client to send jobs to the Compute Tasks API.|

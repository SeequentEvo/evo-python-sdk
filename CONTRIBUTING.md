# Contributing

Thanks for your interest in contributing to Seequent software. Everyone is welcome to contribute to our software!
There are many ways you can contribute. This document provides a summary of some of these ways.

## Asking a question

If you have a question about how the code in this library works, or would like to propose a change, feel free to
open a new issue on GitHub.

If you have a general question, head over to the Evo Group in the [Seequent Community](https://community.seequent.com/group/19-evo/).

## Reporting bugs/issues

Bug reports and feature requests are welcomed in the form of GitHub issues.

You can search existing issues to see if others have already reported your issue. If you find something relevant, you
can react to it using a 👍 or 👎 emoji. Please avoid "+1" or "me too" style comments on the issue without providing
further context or helpful information.

If you can't find an existing issue that represents your request, open a new issue. When opening issues, ensure you
provide a clear description of the problem or idea, along with any necessary context (software version, operating
system, example code, steps to reproduce, etc.). We also welcome pull requests to fix bugs instead of opening issues!

## Opening a pull request

We welcome all forms of pull requests, and strive to ensure contributions are reviewed and merged promptly.

Seequent requires that all commits are signed with verified signatures. Please ensure you configure commit signing before creating a pull request. See [the GitHub documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification) for more information.

### Adding a new project
A project is defined as a self-contained piece of functionality. Each new project is in a sub-folder of the `./packages/` folder in this repository. The folder name should match the package name that will be published and contain the source code for that project.

Because each project is self-contained, contributors must specify maintainers for each new package. These maintainers are responsible for reviewing pull requests, ensuring code quality, maintaining security standards, and providing ongoing project maintenance. Add an entry to the [CODEOWNERS file](.github/CODEOWNERS) in the root directory, for example:

```
# Package maintainers for the new project
packages/evo-mypackage/  @SeequentEvo/mypackage-maintainers
```

Where possible, assign code ownership to a team rather than individuals.
Remember that more specific rules override general ones, so package-specific entries will take precedence over the global fallback rule.

### Publishing a new release
See [RELEASE.md](RELEASE.md) for instructions on how to publish a new GitHub and PyPi release of a package.

### Checklist

To ensure your pull request is merged as quickly as possible, please consider the following:

* Try to prevent breaking changes and ensure backwards compatibility. If a breaking change is necessary, please call
  them out in your pull request.
* Reference issues in your pull request if you're closing one.
* Ensure your code has been automatically linted.
* Verify that all tests pass, and write new tests with appropriate code coverage for new code.
* Verify that all sample code and example notebooks can be run successfully.

### Contributor License Agreement (CLA)

The first time you make a pull request, you will be required to sign a Contributor License Agreement (CLA). This is
managed and tracked by a bot, and only needs to be done once per repository.

The CLA ensures you retain copyright to your contributions, and provides us the right to use, modify, and redistribute
your contributions using the Apache 2.0 License.

## Code of conduct

To ensure an inclusive community, contributors and users in the Seequent community should follow
[the code of conduct](CODE_OF_CONDUCT.md).

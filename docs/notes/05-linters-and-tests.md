# Linters and Tests

## Linters

A linter is a tool that analyzes source code and provides feedback on potential errors, coding style violations, and programming best practices. It helps maintain code quality and consistency by enforcing coding standards and identifying potential bugs or issues early in the development process.

### Benefits of Linters

- **Improved Code Quality**: Linters help catch common programming mistakes, maintain consistent coding style, and enforce best practices, leading to cleaner and more reliable code.

- **Early Bug Detection**: By analyzing code statically, linters can identify potential issues before the code is executed, enabling early bug detection and reducing the likelihood of bugs reaching production.

- **Readability and Maintainability**: Linters promote consistent coding style and naming conventions, making the code more readable and maintainable. This improves collaboration among team members and makes it easier to understand and modify the codebase.

- **Codebase Standardization**: Linters enforce coding standards and guidelines across the entire codebase, ensuring a unified coding style and reducing the risk of introducing code inconsistencies or errors.

### Linters Requirements

We require the following linters to be run on the code and included as checks in continuous integration. See the [[dev] optional dependencies](02-packaging.md#2-projectoptional-dependencies) section of the packaging note for a concise description of these.

- [black](https://black.readthedocs.io/en/stable/)
- [flake8](https://flake8.pycqa.org/en/latest/): Run command `flake8 src/ccn-template/ --max-complexity 10`
- [isort](https://pycqa.github.io/isort/)
- [pydocstyle](http://www.pydocstyle.org/en/latest/)

!!! warning "Beware of Auto-generated Scripts"
    Auto-generated scripts may not adhere to linting standards, leading to failures. For instance, the `version.py` script generated by `setuptools-scm` ([packaging note](02-packaging.md)) doesn't comply with `black` linting standards and doesn't follow the numpy conventions, resulting in `black` and `pydocstyle` failures.

    To circumvent this, you can exclude the problematic file:

    - **Exclude problematic files, e.g., `version.py`**: Update the `pyproject.toml` configuration to skip such files during linting:
      ```toml
      [tools.black]
      # other configurations...
      exclude =  "/version.py$"

      [tools.pydocstyle]
      # other configurations...
      match =  "^(?!version\\.py$).*\\.py$" # Matches all .py files except version.py.
      ```

## Tests

Tests are an essential part of the software development process. They verify the correctness and expected behavior of 
the code, ensuring that it performs as intended. Tests help identify bugs, ensure code quality, and provide confidence 
in the reliability of the software.

### Benefits of Tests

- **Bug Detection**: Tests help identify bugs early in the development cycle, allowing developers to fix them before they reach production. By executing various scenarios, edge cases, and boundary conditions, tests increase the likelihood of discovering potential issues.

- **Regression Prevention**: When modifications are made to the codebase, tests act as a safety net by ensuring that existing functionality remains intact. Tests catch regressions, preventing the reintroduction of bugs when new features or changes are introduced.

- **Code Documentation**: Test cases act as a form of documentation, illustrating how different components of the codebase should behave and providing real-world examples of their usage. This helps improve code comprehension and assists developers in understanding how to interact with the code.

- **Codebase Maintainability**: Tests promote codebase maintainability by facilitating refactoring and code evolution. They provide confidence that changes do not introduce unintended side effects or break existing functionality, enabling developers to make improvements with ease.

### Standard Tests and Use Cases

Below, we provide a non-comprehensive list of test types along with their general use cases and some examples.

#### Unit Testing:

Unit testing aims to verify the functionality of code in isolation.

- **Target:** Individual functions or methods.

- **Goal:** Ensure that the function/method works as expected, such as accepting specific inputs, raising exceptions 
when required, and returning specific outputs.

- **Example:** Test a function that calculates the mean value of a given array. Prepare an input array 
`[1, 2, 3, 4, 5]`, compute the mean by hand (which is 3), apply your function to the array, and verify that the output 
matches the calculated mean.

#### Integration Testing:

Integration testing is performed to validate the interaction and communication between different components or modules of a package.

- **Target:** Interactions between code modules.

- **Goal:** Check if the integrated components work together as expected and identify any issues that may arise due to 
the integration.

- **Example:** Test the compatibility between a function that calculates and outputs the mean of two given vectors and 
another function that computes the ratio of two numbers. Create two arrays, `x = [1, 2, 3, 4, 5]` and 
`y = [2, 4, 6, 8, 10]`. Compute the ratio of the means of `x` and `y` (which is 0.5) and check that the result of
composing your functions equals 0.5.

#### Functional Testing:

Functional testing verifies that the software meets the specified functional requirements.

- **Target:** The application or package as a whole.

- **Goal:** Ensure that the application or package performs the intended functions and produces the correct outputs for 
various inputs.

- **Example:** Test a package for numerical optimization. Define an objective function with known minima. Test all the 
minimization routines with multiple initializations and multiple required precision levels. Make sure that the 
algorithms converge, and that the result is always close to a local minimum up to the defined precision.

#### Regression Testing:

Regression testing is conducted to verify that recent changes or modifications to the code-base have not introduced
 new bugs or issues and have not affected existing functionality.

- **Target:** Previously tested features.

- **Goal:** Ensure that the function and modules still work as expected after a change to the code-base.

- **Example:** Setting up a continuous integration workflow that automatically re-run all implemented tests when a pull request is opened.


### Test Requirements

We require testing your code with [pytest](https://docs.pytest.org/en/) and [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/). 
Your test scripts must reside in the `tests/` folder, and all scripts must start with `test_*.py`. 
Functions and classes within the test scripts should start  with `Test` or `test_`. 

Tests should be written as code is written, and they must be included as checks in continuous integration. 
Pull Requests should not be merged unless the new functionality is adequately tested. It is recommended to start testing 
as early as possible.

Tests should include compatibility testing with other packages, even if they're not strict requirements. 
For example, you can check that code objects can be used by scikit-learn's cross-validation procedure.

### Test coverage

Test coverage is a metric used to measure the extent to which the source code of a program is 
exercised by a test suite. It indicates the percentage of code lines, branches, statements, or conditions 
that have been executed during the execution of the tests. You can perform test and coverage analysis by running the command
```terminal
pytest --cov=packagename test/
```

You should aim to have as complete coverage as possible, we recommend at least 70% coverage.

### Test Figure Generation with `matplotlib`

If you are testing visualization tools that generate figures using `matplotlib`, it's important to set the `matplotlib` backend 
to `'agg'` and close all generated figures before your test function returns.

The `'agg'` backend is specifically designed for static figure rendering and does not initiate an event loop like the `'Qt5Agg'` 
or `'TkAgg'` backends. Consequently, it ensures that the execution of `pytest` is not frozen when figures are generated.

Closing all figures guarantees that the test will terminate when the function returns.

Consider the following example to guide your testing script:

```python
import matplotlib
matplotlib.use("agg")

def test_visualization_tools():
    # ... Perform some plotting
    plt.close('all')


```

In the example above, the `matplotlib.use("agg")` statement sets the backend to `'agg'`, ensuring that figures are rendered statically. In the `test_visualization_tools()` function, you can perform your plotting operations and subsequently close all figures using `plt.close('all')`.

By adhering to these practices, you can effectively test figure generation with `matplotlib` while ensuring that your testing process remains uninterrupted and terminates successfully.


### Additional Resources

The topic of function testing is broad and it is an active area of research. Here are some potentially interesting resources:

#### Websites
1. [https://www.fuzzingbook.org/#A-Textbook-for-Paper,-Screen,-and-Keyboard](https://www.fuzzingbook.org/#A-Textbook-for-Paper,-Screen,-and-Keyboard)
2. [https://www.softwaretestinghelp.com/](https://www.softwaretestinghelp.com/)

#### Books
1. ["Clean Code: A Handbook of Agile Software Craftsmanship" by Robert C. Martin](https://www.goodreads.com/book/show/3735293-clean-code)
2. ["Test Driven Development: By Example" by Kent Beck](https://www.goodreads.com/book/show/387190.Test_Driven_Development)

## Running Tests and Linters with Tox

We recommend using [tox](https://tox.wiki/en/latest/) to run tests and linters. Tox is a command-line tool that automates and simplifies testing and development workflows in Python projects. It provides a consistent and reproducible environment for testing and running code across multiple configurations.

### Benefits of Using Tox

- **Reproducible Testing**: Tox enables developers to define and run tests consistently across different platforms and environments. This ensures that tests produce consistent results and helps catch potential issues that might arise on specific configurations.

- **Continuous Integration (CI) Support**: Tox is commonly used in CI systems, such as Jenkins, Travis CI, and GitHub Actions. It facilitates easy integration with CI pipelines, allowing for automated testing and continuous delivery.

- **Dependency Management**: By encapsulating dependencies within each test environment, Tox helps manage and isolate dependencies, reducing conflicts and ensuring that tests run in the intended environment.

- **Code Quality and Coverage**: Tox can be configured to run additional tools like code linters, static analyzers, and test coverage tools. This helps maintain code quality and ensure sufficient test coverage.

### Usage

#### Installation

Install `tox` from pypi: `pip install tox`.

#### Configuration

To configure Tox, edit the tox.ini file located in the main repository directory. You can use the provided tox.ini 
file in this template repository as a starting point for setting up your tests and linters.

Here is a description of the initialization settings:

##### 1. [tox]

- `isolated_build = True`: Enables isolated build mode, which creates separate virtual environments for each test environment. This helps ensure clean and isolated testing environments.
- `envlist = env1,env2,...`: Specifies the names of test environments to be executed.

##### 2. [testenv]

- `extras = dev`: The tox environment will be installed with `pip install .[dev]`, installing the optional `dev` dependencies specified in `pyproject.toml` section `[project.optional-dependencies]`. By using this (instead of `deps`), we allow the individual tests / linters to be run either separately or within tox and avoid having to list them in both `tox.ini` and `pyproject.toml`. See the [packaging note](02-packaging.md) for some more details on optional dependencies.
    
- `commands`: Specifies the commands to be executed for testing and linting within each test environment.

##### 3. [tox.ini]

- `[gh-actions]`: Provides configuration for GitHub Actions.
    - `python`: Maps the Python versions specified in GitHub Actions to the corresponding test environments in `envlist`.
- `[flake8]`: Configures the flake8 linter.
    - `max-complexity = `: Sets the maximum complexity threshold for code.
    - `max-line-length = `: Sets the maximum line length for code.
    - `exclude`: Specifies the file and directory patterns to be excluded from linting.

##### 4. [testenv:envname]

- `deps`: dependency that are specific to `envname`. `envname` must be listed in `envlist`

- `commands`: commands run only in `envname`. `envname` must be listed in `envlist`

### Running tox

To run tests and linters using Tox, execute the following command within your Python environment from the repository base directory:

```terminal
tox -e py
```

The -e py option sets the test environment name according to the Python version of your environment. 
If Python 3.X is installed in your environment, the test environment name will be set to `py3X`. 
The `py3X` environment must be listed in the `envlist` specified under `[tox]` in the `tox.ini` file.
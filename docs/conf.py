# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import smolgp

extensions = [
    "autoapi.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.lightbox2",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
math_numfig = True  # for section-numbered equations
master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Jupyter -------------------------------------------------
# nb_execution_mode = "auto" # TODO: turn this back on for CI
nb_execution_mode = "off"
nb_execution_excludepatterns = ["benchmarks.ipynb"]
nb_execution_timeout = -1

# -- AutoAPI -------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../src"]
# autodoc_output_dir = "apidocs"
# autoapi_add_toctree_entry = True
# autodoc_type_aliases = {
#     "JAXArray": "tinygp.helpers.JAXArray",
# }

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "smolgp"
copyright = "2025, 2026 Ryan Rubenzahl, Soichiro Hattori, Simons Foundation, Inc."
author = "Ryan Rubenzahl & Soichiro Hattori"
# version = smolgp.__version__
# release = smolgp.__version__
version = "0.0.1"
release = "0.0.1"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "smolgp"
html_logo = "_static/smolgp-logo.png"
html_favicon = "_static/favicon.png"
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/smolgp-dev/smolgp/",
    "repository_branch": "main",
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "notebook_interface": "jupyterlab",
    #     "colab_url": "https://colab.research.google.com/",
    # },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://smolgp.readthedocs.io/en/latest/"

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "GaMPEN"
copyright = "2022, Aritra Ghosh, Amrit Rau, Aayush Mishra"
author = "Aritra Ghosh, Amrit Rau, Aayush Mishra"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "strikethrough",
]

myst_heading_anchors = (
    5  # tells the myst_parser to add anchors to all headings
)
# of level 5 and above

copybutton_exclude = (
    ".linenos, .gp"  # exclude line numbers and the prompt from the copied code
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
autoapi_dirs = ["../../ggt"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

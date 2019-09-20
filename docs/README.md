This directory holds the source files required to build the Edge TPU reference with Sphinx.

You can build these docs locally with the `docs` make target. Of course, this requires that
you install Sphinx and other Python dependencies:

    # We require Python3, so if that's not your default, first start a virtual environment:
    python3 -m venv ~/.my_venvs/coraldocs
    source ~/.my_venvs/coraldocs/bin/activate

    # Navigate to the edgetpu/ directory (parent of docs/) and run these commands...

    # Install the doc build dependencies:
    pip install -r docs/requirements.txt

    # Build the docs for local viewing (in "read the docs" style):
    sphinx-build -b html docs/ docs/_build/html -D html_theme="sphinx_rtd_theme" -D html_file_suffix=".html" -D html_link_suffix=".html"

    # Build the docs for the coral website (with .md files)
    sphinx-build -b html docs/ docs/_build/html

    # Clean the output for a fresh build:
    rm -rf docs/_build

The results are output in `python-tflite-source/docs/_build/html/`.

If the `sphinx-build` command spits errors about the `edgetpu.swig.edgetpu_cpp_wrapper` module not
being found, then you probably need to build that by running `edgetpu/scripts/build_swig.sh`.

For more information about the syntax in these RST files, see the [reStructuredText documentation](
http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).


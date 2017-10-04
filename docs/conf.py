# conf.py
import sys
import os.path

# Remove the default attribute documentation because it will always be None
from sphinx.ext.autodoc import ClassLevelDocumenter, InstanceAttributeDocumenter

def iad_add_directive_header(self, sig):
    ClassLevelDocumenter.add_directive_header(self, sig)


InstanceAttributeDocumenter.add_directive_header = iad_add_directive_header


# Add autodoc and napoleon to the extensions list
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

master_doc = 'index'
html_theme = "sphinx_rtd_theme"

sys.path.insert(0, os.path.abspath(".."))

copyright = 'D-Wave Systems Inc.'

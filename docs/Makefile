# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

liveserve: clean
	@echo "Building and serving documentation with live reload..."
	@echo "Watch for changes in: $(SOURCEDIR)"
	@echo "Access at: http://localhost:8001"
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--port 8001 \
		--watch "$(SOURCEDIR)" \
		--ignore "$(BUILDDIR)" \
		--delay 1 \
		--open-browser

clean:
	rm -rf "$(BUILDDIR)"

.PHONY: help liveserve clean Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

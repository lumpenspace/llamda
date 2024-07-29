#!/bin/bash
pytest tests -vv > DEBUG
pytest --cov=llamda_fn > COV
cd docs
make text
cd ..
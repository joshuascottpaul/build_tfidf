# Changelog

## 0.1.0
- Initial scaffolding.
## 0.1.5
- Pin openai to 1.61.0 to avoid jiter Rust build in Homebrew.
- Keep faiss-cpu pinned to 1.10.0 for PyPI availability.
- Add dependency pin rationale to README.

## 0.1.6
- Add explicit transitive runtime pins for Homebrew installs.
- Document relocation skip for tiktoken.

## 0.1.7
- Add query helpers for open, reveal, and clipboard.
- Add paths-only output and default file dedupe.

## 0.1.8
- Add exceptiongroup to runtime deps for Homebrew on Python 3.10.

## 0.1.9
- Show help on no args.
- Fix shorthand query parsing.

# Streamlit Runtime Config

- `config.toml` defines local Streamlit runtime defaults for this repository.
- `fileWatcherType = "poll"` is used to avoid inotify watch-limit failures on shared servers.

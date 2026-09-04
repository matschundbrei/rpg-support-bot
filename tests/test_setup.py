import yaml

from rpg_bot.cli.setup import build_config_yaml, build_env_text, mask_key


def _cfg() -> dict:
    return {
        "llm": {
            "backend": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "",
            "max_tokens": 4096,
            "temperature": 0.3,
            "max_history": 20,
        },
        "embeddings": {
            "model": "nomic-embed-text-v2-moe:latest",
            "base_url": "http://localhost:11434/v1",
        },
        "chunking": {"chunk_size": 1500, "chunk_overlap": 200},
        "retrieval": {"top_k": 15, "relevance_threshold": 1.0},
        "chromadb": {"persist_directory": "data/chromadb", "collection_name": "rpg_sourcebooks"},
        "sourcebooks_directory": "sourcebooks",
    }


def test_build_config_yaml_roundtrip():
    parsed = yaml.safe_load(build_config_yaml(_cfg()))
    assert parsed["llm"]["backend"] == "anthropic"
    assert parsed["llm"]["model"] == "claude-sonnet-4-6"
    assert parsed["llm"]["max_tokens"] == 4096
    assert parsed["llm"]["temperature"] == 0.3
    assert parsed["embeddings"]["model"] == "nomic-embed-text-v2-moe:latest"
    assert parsed["embeddings"]["base_url"] == "http://localhost:11434/v1"
    assert parsed["chunking"] == {"chunk_size": 1500, "chunk_overlap": 200}
    assert parsed["retrieval"]["top_k"] == 15
    assert parsed["retrieval"]["relevance_threshold"] == 1.0
    assert parsed["chromadb"]["persist_directory"] == "data/chromadb"
    assert parsed["sourcebooks_directory"] == "sourcebooks"


def test_build_config_yaml_openai_base_url():
    cfg = _cfg()
    cfg["llm"]["backend"] = "openai"
    cfg["llm"]["base_url"] = "http://localhost:1234/v1"
    parsed = yaml.safe_load(build_config_yaml(cfg))
    assert parsed["llm"]["backend"] == "openai"
    assert parsed["llm"]["base_url"] == "http://localhost:1234/v1"


def test_build_env_text():
    text = build_env_text({"ANTHROPIC_API_KEY": "sk-ant-123", "OPENAI_API_KEY": "", "API_KEY": ""})
    assert "ANTHROPIC_API_KEY=sk-ant-123" in text
    assert "OPENAI_API_KEY=" in text
    assert "API_KEY=" in text


def test_mask_key():
    assert mask_key("") == "(empty)"
    assert mask_key("short") == "(set)"
    assert mask_key("sk-ant-abcdef123456") == "sk-a...3456"


def test_run_setup_writes_default_config(tmp_path, monkeypatch):
    import rpg_bot.cli.setup as setup_mod

    monkeypatch.setattr(setup_mod, "_PROJECT_ROOT", tmp_path)

    answers = [""] * 14 + ["y"]
    monkeypatch.setattr("builtins.input", lambda prompt="": answers.pop(0) if answers else "")
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")

    setup_mod.run_setup()

    parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert parsed["llm"]["backend"] == "anthropic"
    assert parsed["chunking"]["chunk_size"] == 1500
    env = (tmp_path / ".env").read_text()
    assert "ANTHROPIC_API_KEY=" in env


def test_run_setup_aborts_without_writing(tmp_path, monkeypatch):
    import rpg_bot.cli.setup as setup_mod

    monkeypatch.setattr(setup_mod, "_PROJECT_ROOT", tmp_path)

    answers = [""] * 14 + ["n"]
    monkeypatch.setattr("builtins.input", lambda prompt="": answers.pop(0) if answers else "")
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")

    setup_mod.run_setup()

    assert not (tmp_path / "config.yaml").exists()
    assert not (tmp_path / ".env").exists()

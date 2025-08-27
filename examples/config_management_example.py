#!/usr/bin/env python3
"""
Example of using the new workdir configuration management system.

This example demonstrates:
1. Setting a custom workdir
2. Modifying configuration
3. Saving configuration to workdir
4. Loading configuration from workdir
5. Using environment variables
"""

import os
from pathlib import Path
from agraph.config import (
    get_settings,
    save_config_to_workdir,
    load_config_from_workdir,
    has_workdir_config,
    update_settings,
    reset_settings
)

def main():
    print("=== AGraph Configuration Management Example ===\n")

    # Example 1: Using environment variable to set workdir
    print("1. Setting up custom workdir using environment variable:")
    project_root = Path(__file__).parent.parent
    custom_workdir = str(project_root / "example_workdir")
    os.environ["AGRAPH_WORKDIR"] = custom_workdir

    # Reset to pick up the new workdir
    reset_settings()

    settings = get_settings()
    print(f"   Current workdir: {settings.workdir}")
    print(f"   Config exists in workdir: {has_workdir_config()}")

    # Example 2: Configure for different LLM providers
    print("\n2. Configuring for different scenarios:")

    # Configuration for development with OpenAI
    dev_config = {
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 4096
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimension": 1536
        },
        "text": {
            "max_chunk_size": 512,
            "chunk_overlap": 100
        }
    }

    print("   Applying development configuration...")
    update_settings(dev_config)

    current = get_settings()
    print(f"   LLM Provider: {current.llm.provider}")
    print(f"   LLM Model: {current.llm.model}")
    print(f"   Temperature: {current.llm.temperature}")
    print(f"   Embedding Model: {current.embedding.model}")

    # Example 3: Save configuration to workdir
    print("\n3. Saving configuration to workdir:")
    config_path = save_config_to_workdir()
    print(f"   Configuration saved to: {config_path}")

    # Example 4: Load configuration from workdir
    print("\n4. Testing configuration persistence:")

    # Simulate restart by resetting settings
    reset_settings()

    # Load the saved configuration
    loaded_settings = load_config_from_workdir()
    if loaded_settings:
        print("   ✓ Configuration loaded successfully from workdir")
        print(f"   LLM Model: {loaded_settings.llm.model}")
        print(f"   Temperature: {loaded_settings.llm.temperature}")
    else:
        print("   ✗ Failed to load configuration")

    # Example 5: Different configurations for different environments
    print("\n5. Managing multiple configurations:")

    # Production configuration
    prod_config = {
        "llm": {
            "model": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 8192
        },
        "text": {
            "max_chunk_size": 1024,
            "chunk_overlap": 200
        }
    }

    update_settings(prod_config)
    prod_config_path = save_config_to_workdir("production.json")
    print(f"   Production config saved to: {prod_config_path}")

    # Testing configuration
    test_config = {
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 2048
        },
        "text": {
            "max_chunk_size": 256,
            "chunk_overlap": 50
        }
    }

    update_settings(test_config)
    test_config_path = save_config_to_workdir("testing.json")
    print(f"   Testing config saved to: {test_config_path}")

    # Example 6: Loading specific configurations
    print("\n6. Loading specific configurations:")

    # Load production config
    prod_settings = load_config_from_workdir("production.json")
    if prod_settings:
        print(f"   Production - Model: {prod_settings.llm.model}, Max Tokens: {prod_settings.llm.max_tokens}")

    # Load testing config
    test_settings = load_config_from_workdir("testing.json")
    if test_settings:
        print(f"   Testing - Model: {test_settings.llm.model}, Max Tokens: {test_settings.llm.max_tokens}")

    print("\n=== Example completed successfully! ===")
    print(f"\nYou can find the configuration files in: {custom_workdir}")
    print("To use a specific configuration, set AGRAPH_WORKDIR environment variable")
    print("and place your config.json in that directory.")

if __name__ == "__main__":
    main()

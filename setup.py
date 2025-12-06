import os

project_structure = {
    "": {
        "README.md": None,
        "requirements.txt": None,
        ".dockerignore":None,
        "docker-compose.yaml":None,
        ".gitignore": None,
        ".env.example": None,
        
        "src": {
            "rag": {
                "rag_langchain.py": None,
                "ingestion.py": None,
                "retriever.py": None,
                "fact_check.py": None,
            },
            "agents": {
                "writer_agent.py": None,
                "fact_checker_agent.py": None,
                "outline_agent.py":None,
                "tone_editor.py": None,
            },
            "tools":{
                "fact_checker_tools.py":None,
                "outline_tools.py":None,
                "writer_tools.py":None,
            },
            "utils": {
                "logger.py": None,
                "constants.py": None,
                "prompts.py": None,
                "config.py":None,
            },
            "main.py": None,
        },

        "notebooks": {
            "inspect_corpus.ipynb": None
        },

        "scripts": {
            "run_rag.sh": None,
            "build_index.py": None,
            "evaluate.py": None,
        },

        "tests": {
            "test_rag.py": None,
            "test_agents.py": None,
            "test_factcheck.py": None,
        }
    }
}


def create_structure(base_path, structure):
    """Recursively create folders & files based on provided dict."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        # If content is a dict → it's a folder
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # It's a file → create placeholder
            folder = base_path
            file_path = os.path.join(folder, name)

            with open(file_path, "w", encoding="utf-8") as f:
                if name.endswith(".py"):
                    f.write(f"# {name}\n# Auto-generated placeholder\n\n")
                elif name.endswith(".md"):
                    f.write(f"# {name.replace('.md','').replace('_',' ').title()}\n\n")
                elif name.endswith(".txt"):
                    f.write("# Prompt Template Placeholder\n\n")
                elif name.endswith(".json"):
                    f.write("{\n  \n}\n")
                elif name.endswith(".sh"):
                    f.write("#!/bin/bash\n# Auto-generated script\n\n")
                elif name.endswith(".ipynb"):
                    f.write("")  # Leave blank; user can populate in VS Code
                else:
                    f.write("")  # generic empty file


if __name__ == "__main__":
    root = os.getcwd()
    print("\nCreating Kerala Ayurveda AI folder structure...\n")
    create_structure(root, project_structure)
    print("Project structure created successfully at:")
    print(os.path.join(root, "kerala-ayurveda-ai"))
"""
Setup script for RAG-Anything project.
"""
import os
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project structure."""
    
    print("🚀 Setting up RAG-Anything project structure...")
    
    # Define directory structure
    directories = [
        "storage",
        "storage/lightrag_storage", 
        "storage/processed_docs",
        "storage/cache",
        "documents",
        "documents/pending",
        "documents/processed",
        "logs",
        "examples"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        shutil.copy(".env.example", ".env")
        print("✅ Created .env file from .env.example")
        print("⚠️  Please edit .env file and add your OpenAI API key!")
    else:
        print("ℹ️  .env file already exists")
    
    # Create .gitignore
    gitignore_content = """
# Environment variables
.env

# Storage directories
storage/
logs/
documents/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    print("✅ Created .gitignore file")
    
    print("\n🎉 Project structure created successfully!")
    print("\n📝 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Add PDF files to ./documents/pending/")
    print("4. Run: python main.py interactive")

def install_dependencies():
    """Install required dependencies."""
    import subprocess
    import sys
    
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run manually: pip install -r requirements.txt")

def check_mineru_installation():
    """Check if MinerU is properly installed."""
    try:
        import subprocess
        result = subprocess.run(["mineru", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ MinerU is installed and working")
            return True
        else:
            print("❌ MinerU command not found")
            return False
    except Exception as e:
        print(f"❌ Error checking MinerU: {e}")
        return False

def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check for optional dependencies
    optional_deps = {
        "PIL": "Pillow",
        "reportlab": "reportlab"
    }
    
    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"⚠️  {package} not found (optional for enhanced features)")
    
    # Check MinerU
    if not check_mineru_installation():
        print("⚠️  MinerU not found. Install with: pip install raganything")
    
    return True

def check_lightrag_fix():
    """Check if LightRAG initialization fix works."""
    print("🧪 Testing LightRAG initialization fix...")
    
    try:
        # Test the import
        from lightrag.kg.shared_storage import initialize_pipeline_status
        print("✅ initialize_pipeline_status import successful")
        
        # Test basic initialization (without API calls)
        from lightrag import LightRAG
        print("✅ LightRAG import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install --upgrade lightrag-hku")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎯 RAG-Anything Project Setup")
    print("=" * 40)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Check LightRAG fix
    if not check_lightrag_fix():
        print("❌ LightRAG initialization test failed")
        return
    
    # Create project structure
    create_project_structure()
    
    # Offer to install dependencies
    response = input("\n❓ Install dependencies now? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        install_dependencies()
    
    print("\n🎉 Setup complete!")
    print("\n🚀 Quick start:")
    print("  1. Edit .env file with your OpenAI API key")
    print("  2. python quick_fix_test.py    # Test the fix")
    print("  3. python main.py interactive    # Start interactive mode")
    print("  4. python examples/simple_query.py    # Run simple example")

if __name__ == "__main__":
    main()
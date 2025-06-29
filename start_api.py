"""
Quick start script for RAG-Anything API server.
"""
import os
import webbrowser
import time
from pathlib import Path

def setup_static_files():
    """Ensure static directory and files exist."""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # The index.html file should already be created by the artifacts
    index_file = static_dir / "index.html"
    if not index_file.exists():
        print("⚠️ Web interface file not found. Creating basic interface...")
        # Create a simple placeholder
        index_file.write_text("""
<!DOCTYPE html>
<html>
<head><title>RAG-Anything</title></head>
<body>
    <h1>RAG-Anything API</h1>
    <p>API is running! Visit <a href="/docs">/docs</a> for interactive documentation.</p>
</body>
</html>
        """)

def main():
    print("🚀 RAG-Anything API Quick Start")
    print("=" * 40)
    
    # Setup static files
    setup_static_files()
    
    # Check dependencies and install if missing
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies found")
    except ImportError:
        print("❌ Missing FastAPI dependencies. Installing...")
        import subprocess
        import sys
        
        # Install required packages
        packages = ["fastapi", "uvicorn", "python-multipart"]
        for package in packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        print("✅ Dependencies installed successfully!")
    
    # Start server
    print("\n🌐 Starting API server...")
    print("📡 Server: http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/docs") 
    print("🖥️ Web Interface: http://127.0.0.1:8000")
    print("\n⚡ Press Ctrl+C to stop the server")
    
    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except:
            pass  # Browser opening is optional
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start the server
    import subprocess
    import sys
    try:
        subprocess.run([sys.executable, "api_server.py", "--host", "127.0.0.1", "--port", "8000"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\n💡 Try running manually:")
        print("   pip install fastapi uvicorn python-multipart")
        print("   python api_server.py")

if __name__ == "__main__":
    main()
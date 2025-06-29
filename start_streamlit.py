"""
Start Streamlit interface for RAG-Anything.
"""
import subprocess
import sys
import time
import webbrowser
import threading

def install_streamlit():
    """Install Streamlit if not available."""
    try:
        import streamlit
        print("✅ Streamlit already installed")
        return True
    except ImportError:
        print("📦 Installing Streamlit...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "pandas"])
            print("✅ Streamlit installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            print("💡 Try manually: pip install streamlit pandas")
            return False

def check_api_server():
    """Check if API server is running."""
    import requests
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start API server in background."""
    print("🚀 Starting API server...")
    subprocess.Popen([sys.executable, "api_server.py"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    
    # Wait for server to start
    for i in range(10):
        if check_api_server():
            print("✅ API server started successfully")
            return True
        time.sleep(1)
    
    print("⚠️ API server may still be starting...")
    return False

def open_browser():
    """Open browser after delay."""
    time.sleep(3)
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass

def main():
    """Main startup function."""
    print("🌟 RAG-Anything Streamlit Interface")
    print("=" * 40)
    
    # 1. Install Streamlit
    if not install_streamlit():
        return
    
    # 2. Check API server
    if not check_api_server():
        print("🔍 API server not running, starting it...")
        start_api_server()
    else:
        print("✅ API server already running")
    
    # 3. Start browser opener thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 4. Start Streamlit
    print("\n🎉 Starting Streamlit interface...")
    print("📱 Interface: http://localhost:8501")
    print("📡 API Server: http://127.0.0.1:8000")
    print("\n⚡ Press Ctrl+C to stop")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        print("💡 Try manually:")
        print("   pip install streamlit")
        print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
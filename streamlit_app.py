import app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8501))  # Default 8501 locally
    st.experimental_singleton.clear()  # Optional: clear singletons on restart
    import streamlit.web.bootstrap
    streamlit.web.bootstrap.run(
        f"{__file__}", 
        command_line=None, 
        args=["--server.port", str(port), "--server.address", "0.0.0.0"], 
        _is_running_with_streamlit=True
    )

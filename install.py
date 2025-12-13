"""
Installation script for ComfyUI Text-to-Pose nodes.
This script is automatically executed by ComfyUI Manager during installation.
"""

import subprocess
import sys
import os


def install():
    """
    Called by ComfyUI Manager during installation.
    Clones the text-to-pose repository and sets up the t2p module.
    """
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    t2p_repo_path = os.path.join(repo_dir, "t2p_repo")
    t2p_link_path = os.path.join(repo_dir, "t2p")
    utils_src = os.path.join(t2p_repo_path, "utils.py")
    utils_dst = os.path.join(repo_dir, "utils.py")

    print("[Text-to-Pose] Starting installation...")

    # Clone t2p repository if not exists
    if not os.path.exists(t2p_repo_path):
        print("[Text-to-Pose] Cloning text-to-pose repository...")
        try:
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/logicalor/text-to-pose",
                    t2p_repo_path
                ],
                check=True,
                capture_output=True,
                text=True
            )
            print("[Text-to-Pose] Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[Text-to-Pose] Error cloning repository: {e.stderr}")
            raise
    else:
        print("[Text-to-Pose] Repository already exists, updating...")
        try:
            subprocess.run(
                ["git", "-C", t2p_repo_path, "pull"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError:
            print("[Text-to-Pose] Warning: Could not update repository.")

    # Create symlink to t2p module (or copy on Windows)
    t2p_source = os.path.join(t2p_repo_path, "t2p")
    t2p_relative_source = os.path.join("t2p_repo", "t2p")  # Relative path for symlink
    
    if os.path.exists(t2p_link_path) or os.path.islink(t2p_link_path):
        if os.path.islink(t2p_link_path):
            os.unlink(t2p_link_path)
        elif os.path.isdir(t2p_link_path):
            import shutil
            shutil.rmtree(t2p_link_path)

    if os.path.exists(t2p_source):
        print("[Text-to-Pose] Setting up t2p module...")
        try:
            # Use relative symlink (works on Linux/Mac and Windows with dev mode)
            os.symlink(t2p_relative_source, t2p_link_path)
            print("[Text-to-Pose] Created relative symlink to t2p module.")
        except OSError:
            # Fallback to copying for Windows without dev mode
            import shutil
            shutil.copytree(t2p_source, t2p_link_path)
            print("[Text-to-Pose] Copied t2p module (symlink not supported).")

    # Copy utils.py for pose drawing functions
    if os.path.exists(utils_src) and not os.path.exists(utils_dst):
        import shutil
        shutil.copy2(utils_src, utils_dst)
        print("[Text-to-Pose] Copied utils.py for pose rendering.")

    print("[Text-to-Pose] Installation complete!")
    print("[Text-to-Pose] Please restart ComfyUI to load the new nodes.")


def uninstall():
    """
    Called by ComfyUI Manager during uninstallation.
    Cleans up cloned repositories and symlinks.
    """
    import shutil
    
    repo_dir = os.path.dirname(os.path.realpath(__file__))
    t2p_repo_path = os.path.join(repo_dir, "t2p_repo")
    t2p_link_path = os.path.join(repo_dir, "t2p")
    utils_path = os.path.join(repo_dir, "utils.py")

    print("[Text-to-Pose] Starting uninstallation...")

    # Remove symlink/copy of t2p module
    if os.path.islink(t2p_link_path):
        os.unlink(t2p_link_path)
        print("[Text-to-Pose] Removed t2p symlink.")
    elif os.path.isdir(t2p_link_path):
        shutil.rmtree(t2p_link_path)
        print("[Text-to-Pose] Removed t2p directory.")

    # Remove cloned repository
    if os.path.exists(t2p_repo_path):
        shutil.rmtree(t2p_repo_path)
        print("[Text-to-Pose] Removed cloned repository.")

    # Remove utils.py
    if os.path.exists(utils_path):
        os.remove(utils_path)
        print("[Text-to-Pose] Removed utils.py.")

    print("[Text-to-Pose] Uninstallation complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall()
    else:
        install()

"""User-operated local masked input, never chat or repository literals."""


def configure(path):
    import json
    import os
    import tkinter as tk
    from pathlib import Path

    path = Path(path)
    initial = json.loads(path.read_text()) if path.exists() else {}
    root = tk.Tk()
    root.title("eBay local credentials")
    root.geometry("620x300")
    tk.Label(root, text="Enter credentials locally. Secrets are not printed or sent to chat.").pack(
        pady=12
    )
    env = tk.StringVar(value=initial.get("environment", "sandbox"))
    tk.OptionMenu(root, env, "sandbox", "production").pack()
    tk.Label(root, text="App ID / Client ID").pack()
    client = tk.Entry(root, width=65)
    client.insert(0, initial.get("client_id", ""))
    client.pack()
    tk.Label(root, text="Cert ID / Client Secret").pack()
    secret = tk.Entry(root, width=65, show="*")
    secret.pack()
    saved = []

    def save():
        if not client.get().strip() or not secret.get().strip():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "environment": env.get(),
                    "client_id": client.get().strip(),
                    "client_secret": secret.get().strip(),
                },
                f,
            )
        os.chmod(path, 0o600)
        saved.append(True)
        root.destroy()

    tk.Button(root, text="Save locally", command=save).pack(pady=20)
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()
    if not saved:
        raise ValueError("credential entry cancelled")

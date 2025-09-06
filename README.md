## Install via Blender Extensions (Remote Repository)

Blender 4.2+ can install and auto-update extensions from a static repository.

Add this repository in Blender:

- Open Blender → Get Extensions → Repositories → [+] Add Remote Repository
- URL: `https://<your-username>.github.io/<your-repo>/index.json`
- Save. The extension appears in the Extensions browser and updates automatically.

Drag-and-drop install link (optional):

- Visit: `https://<your-username>.github.io/<your-repo>/extensions.html`
- Drag the “Download ZIP (with metadata)” button into Blender.

Local testing (optional):

- Generate locally: `blender --command extension server-generate --repo-dir /path/to/packages`
- Add URL:
  - Linux/macOS: `file:///path/to/packages/index.json`
  - Windows: `file:///C:/path/to/packages/index.json`

Notes:

- CI builds a ZIP on push to `main` and publishes a rolling `latest` release.
- Pages workflow downloads that ZIP, generates `docs/index.json`, and publishes `docs/`.

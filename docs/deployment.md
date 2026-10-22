# Deployment assessment

Decision: keep this version local for learning and portfolio demonstrations. No service
has been provisioned or published. A recorded synthetic demo and the evaluation report
are enough to present the project today.

| Option | Fit | Work needed |
| --- | --- | --- |
| Local Streamlit | Current recommended demo | Local model cache, `.env`, persistent data directory |
| Streamlit Community Cloud | Potential small synthetic demo | GitHub app configuration, hosted secrets, resource/storage validation, restrict mutations |
| Container or VM with persistent storage | Potential persistent private app | Durable volume, deployment packaging, access controls, operational testing |

Community Cloud deploys an app from a GitHub repository and supports secrets through
app settings. Keep `.env`, `.streamlit/secrets.toml`, and uploaded documents out of Git.
Root-level Streamlit secrets can supply environment variables, which this app reads.
See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
Do not assume the current local Chroma setup is durable on an arbitrary hosted filesystem.

For example, Render's normal service filesystem is ephemeral; persistence requires a
persistent disk, with writes under its mount path. Its documented disk constraints must
be checked before selecting a service. Point `RAG_DATA_DIR` at durable storage and run
one writer process for this embedded Chroma design. See [Render persistent disks](https://render.com/docs/disks).
Pricing and available resources should be checked when deployment is actually requested.

Before offering an interactive public app, choose one concrete access model:

- A read-only synthetic demo: disable upload, URL ingestion, and deletion; cap requests
  and Groq usage. All visitors intentionally share the synthetic library.
- A private document app: add authentication and isolate each user's storage, retrieval,
  source access, and deletion. The current global library is shared across sessions.

Then test model download/cache behavior, cold starts, memory use, persistence after a
restart/redeploy, API failure handling, and concurrent sessions on the chosen host.
These deployment changes are deferred; the current app remains a local single-library
application. Public URL protections already exist, but are not a substitute for user
access controls or provider spending limits.

# Neural Tesla Mintlify Portal

This directory contains a Mintlify documentation portal for the repository.

## Generate the evidence-backed pages

From the repository root:

```bash
python3 scripts/generate_mintlify_portal.py
```

## Validate the generated portal locally

From the repository root:

```bash
python3 scripts/generate_mintlify_portal.py --validate-only
```

## Run Mintlify locally

From inside this directory:

```bash
npm i -g mint
mint dev
```

If you prefer not to install the CLI globally, use:

```bash
npx mint dev
```
